"""Triton gather / scatter helpers for FlashSpec.

Tiling strategy
---------------
A single program handles a contiguous BLOCK_SIZE chunk of the flat token
sequence.  Vocab-dimension scatter is not needed — we gather only the
accepted token IDs and write them contiguously.

See Also
--------
docs/kernels.md : SRAM usage analysis.
flashspec.kernels._reference : Pure-PyTorch reference ``gather_accepted_reference``.
"""

from __future__ import annotations

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

__all__ = ["gather_accepted"]

_GATHER_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
]


@triton.autotune(configs=_GATHER_CONFIGS, key=["N_ELEMENTS"])
@triton.jit
def _gather_accepted_kernel(
    token_ids_ptr: tl.tensor,
    first_rejection_ptr: tl.tensor,
    output_ptr: tl.tensor,
    GAMMA: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton kernel: mask out token IDs at or after the first rejection.

    Programs tile over the flat ``(batch_size * gamma)`` space.
    Each element writes the token ID if its gamma-position is before
    ``first_rejection[batch_idx]``, otherwise writes ``-1``.

    See docs/kernels.md §2.2 for tiling and SRAM analysis.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Determine batch index and gamma position from flat offset.
    batch_idx = offsets // GAMMA
    gamma_pos = offsets % GAMMA

    token_id = tl.load(token_ids_ptr + offsets, mask=mask, other=-1)
    first_rej = tl.load(first_rejection_ptr + batch_idx, mask=mask, other=0)

    # Keep token if gamma_pos < first_rejection; else write -1.
    valid = gamma_pos < first_rej
    output = tl.where(valid, token_id, -1)
    tl.store(output_ptr + offsets, output, mask=mask)


def gather_accepted(
    token_ids: torch.Tensor,
    first_rejection: torch.Tensor,
) -> torch.Tensor:
    """Gather accepted tokens up to the first rejection position.

    Parameters
    ----------
    token_ids : torch.Tensor
        Draft token IDs.  Shape: ``(batch_size, gamma)``, dtype int64.
    first_rejection : torch.Tensor
        Index of first rejection per sequence (or ``gamma`` if all accepted).
        Shape: ``(batch_size,)``, dtype int32 or int64.

    Returns
    -------
    torch.Tensor
        Token IDs with positions at or beyond ``first_rejection`` replaced by
        ``-1``.  Shape: ``(batch_size, gamma)``, dtype int64.

    Raises
    ------
    ValueError
        If ``token_ids`` and ``first_rejection`` batch dimensions do not match.
    RuntimeError
        If tensors are not on the same CUDA device.

    Examples
    --------
    >>> ids   = torch.tensor([[10, 20, 30, 40]], device="cuda")
    >>> frej  = torch.tensor([2], dtype=torch.int32, device="cuda")
    >>> gather_accepted(ids, frej)
    tensor([[10, 20, -1, -1]], device='cuda:0')
    """
    batch_size, gamma = token_ids.shape
    if first_rejection.shape != (batch_size,):
        raise ValueError(
            f"first_rejection must have shape ({batch_size},); "
            f"got {first_rejection.shape}."
        )
    if token_ids.device != first_rejection.device:
        raise RuntimeError(
            f"token_ids and first_rejection must be on the same device; "
            f"got {token_ids.device} vs {first_rejection.device}."
        )

    n_elements = batch_size * gamma
    output = torch.empty_like(token_ids)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731

    _gather_accepted_kernel[grid](
        token_ids.contiguous(),
        first_rejection.to(torch.int64).contiguous(),
        output,
        GAMMA=gamma,
        N_ELEMENTS=n_elements,
    )
    return output
