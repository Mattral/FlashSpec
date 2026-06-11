"""Triton token-verification kernel for FlashSpec.

Tiling strategy
---------------
The acceptance test for draft position ``i`` requires only the log-probabilities
at a single vocabulary index (the actual draft token ``draft_token_ids[b, i]``).
We therefore do NOT sweep the full vocab dimension.  Instead:

  * The outer grid is ``(batch_size * gamma,)`` — one program per (batch, draft-pos).
  * Each program reads two scalars (``lp_q``, ``lp_p``), computes the acceptance
    probability, and writes one boolean and one int32 contribution.
  * A second kernel ``first_rejection_kernel`` reduces accepted masks to
    ``first_rejection`` indices using a sequential scan — this is fast because
    ``gamma`` is small (typically 1–8).

See Also
--------
docs/kernels.md : SRAM usage analysis and roofline discussion.
flashspec.kernels._reference : Pure-PyTorch reference (ground truth for tests).
"""

from __future__ import annotations

import torch
import triton  # type: ignore[import]  # triton ships no py.typed; stubs unavailable
import triton.language as tl  # type: ignore[import]  # same as above

__all__ = ["verify_tokens"]

# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------
_VERIFY_CONFIGS = [
    triton.Config({"BLOCK_B": 1}, num_warps=1),
    triton.Config({"BLOCK_B": 2}, num_warps=2),
    triton.Config({"BLOCK_B": 4}, num_warps=4),
    triton.Config({"BLOCK_B": 8}, num_warps=4),
]


# ---------------------------------------------------------------------------
# Acceptance kernel — one program per (batch, draft-position)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_VERIFY_CONFIGS, key=["BATCH_SIZE", "GAMMA"])
@triton.jit
def _verify_acceptance_kernel(
    draft_logprobs_ptr: tl.tensor,
    target_logprobs_ptr: tl.tensor,
    draft_token_ids_ptr: tl.tensor,
    u_ptr: tl.tensor,
    accepted_ptr: tl.tensor,
    BATCH_SIZE: tl.constexpr,
    GAMMA: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
) -> None:
    """Triton kernel: compute acceptance mask for a block of (batch, gamma) pairs.

    Each program handles ``BLOCK_B`` consecutive batch indices for a single
    draft-position index determined by ``tl.program_id(1)``.  The SRAM footprint
    per program is O(BLOCK_B) scalars — constant in VOCAB_SIZE.

    See docs/kernels.md §2.1 for the tiling strategy and SRAM usage analysis.
    """
    batch_block_id = tl.program_id(0)
    gamma_idx = tl.program_id(1)

    batch_offsets = batch_block_id * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = batch_offsets < BATCH_SIZE

    # Flat index into (batch, gamma) shaped tensors
    flat_idx = batch_offsets * GAMMA + gamma_idx

    # Load draft token ids: shape (BLOCK_B,)
    token_ids = tl.load(draft_token_ids_ptr + flat_idx, mask=mask_b, other=0)

    # Compute logprob tensor base offsets for each batch element at this gamma pos
    logprob_base = batch_offsets * GAMMA * VOCAB_SIZE + gamma_idx * VOCAB_SIZE

    # Gather log-probs at the selected token ids
    lp_q = tl.load(
        draft_logprobs_ptr + logprob_base + token_ids, mask=mask_b, other=0.0
    )
    lp_p = tl.load(
        target_logprobs_ptr + logprob_base + token_ids, mask=mask_b, other=0.0
    )

    # Load uniform samples
    u = tl.load(u_ptr + flat_idx, mask=mask_b, other=1.0)

    # Acceptance: u < min(1, exp(log_p - log_q))
    log_ratio = lp_p - lp_q
    accept_prob = tl.minimum(tl.exp(log_ratio.to(tl.float32)), 1.0)
    accepted = u.to(tl.float32) < accept_prob

    tl.store(accepted_ptr + flat_idx, accepted, mask=mask_b)


# ---------------------------------------------------------------------------
# first_rejection reduction (CPU-side, gamma is tiny)
# ---------------------------------------------------------------------------
def _compute_first_rejection(accepted: torch.Tensor) -> torch.Tensor:
    """Compute first-rejection index from boolean accepted mask.

    Parameters
    ----------
    accepted : torch.Tensor
        Boolean mask.  Shape: ``(batch_size, gamma)``.

    Returns
    -------
    torch.Tensor
        Shape: ``(batch_size,)``, dtype int32.  Value is the index of the
        first ``False`` entry, or ``gamma`` if all entries are ``True``.
    """
    batch_size, gamma = accepted.shape
    rejected = ~accepted
    has_rejection = rejected.any(dim=-1)
    first_rej_raw = rejected.int().argmax(dim=-1)
    return torch.where(
        has_rejection,
        first_rej_raw,
        torch.full_like(first_rej_raw, gamma),
    ).to(torch.int32)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------
def verify_tokens(
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    u: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel token acceptance test across all draft positions.

    Implements the acceptance criterion from Algorithm 1 of Leviathan et al.
    (2023) using a Triton kernel.  Operates on pre-computed log-probabilities
    to avoid redundant forward passes.

    Parameters
    ----------
    draft_logprobs : torch.Tensor
        Log-probabilities under the draft model.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
    target_logprobs : torch.Tensor
        Log-probabilities under the target model.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
    u : torch.Tensor
        Uniform samples for acceptance testing.
        Shape: ``(batch_size, gamma)``, dtype float32.  Values in ``[0, 1)``.
    draft_token_ids : torch.Tensor
        Token indices chosen by the draft model.
        Shape: ``(batch_size, gamma)``, dtype int64.

    Returns
    -------
    accepted : torch.Tensor
        Boolean acceptance mask.  Shape: ``(batch_size, gamma)``.
    first_rejection : torch.Tensor
        Index of the first rejection per sequence, or ``gamma`` if all accepted.
        Shape: ``(batch_size,)``, dtype int32.

    Raises
    ------
    ValueError
        If ``draft_logprobs`` and ``target_logprobs`` shapes do not match.
    RuntimeError
        If tensors are not on the same CUDA device.

    Notes
    -----
    The acceptance criterion is ``u_i < exp(log p(x_i) - log q(x_i))``,
    clipped to ``[0, 1]``.  This is numerically equivalent to
    ``u_i < p(x_i) / q(x_i)`` but avoids underflow for low-probability tokens.

    References
    ----------
    .. [1] Leviathan et al. (2023), "Fast Inference from Transformers via
       Speculative Decoding", arXiv:2211.17192, Algorithm 1.

    Examples
    --------
    >>> accepted, first_rejection = verify_tokens(
    ...     draft_logprobs, target_logprobs, u, draft_token_ids
    ... )
    >>> assert accepted.shape == (batch_size, gamma)
    """
    if draft_logprobs.shape != target_logprobs.shape:
        raise ValueError(
            f"draft_logprobs and target_logprobs must have identical shapes; "
            f"got {draft_logprobs.shape} vs {target_logprobs.shape}."
        )
    batch_size, gamma, vocab_size = draft_logprobs.shape
    if u.shape != (batch_size, gamma):
        raise ValueError(
            f"u must have shape ({batch_size}, {gamma}); got {u.shape}."
        )
    if draft_logprobs.device != target_logprobs.device:
        raise RuntimeError(
            f"All tensors must be on the same device; "
            f"got {draft_logprobs.device} vs {target_logprobs.device}."
        )

    accepted = torch.empty((batch_size, gamma), dtype=torch.bool, device=draft_logprobs.device)

    block_b: int = 1  # Autotune selects; this default is only for grid calculation.
    grid = (
        lambda meta: (  # noqa: E731
            triton.cdiv(batch_size, meta["BLOCK_B"]),
            gamma,
        )
    )

    _verify_acceptance_kernel[grid](
        draft_logprobs,
        target_logprobs,
        draft_token_ids,
        u,
        accepted,
        BATCH_SIZE=batch_size,
        GAMMA=gamma,
        VOCAB_SIZE=vocab_size,
    )

    first_rejection = _compute_first_rejection(accepted)
    return accepted, first_rejection
