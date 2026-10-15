"""Standard speculative-sampling acceptance/rejection (Algorithm 1).

This module implements **exactly** Algorithm 1 from Leviathan et al. (2023).
The residual distribution is the same mathematical operation on every
backend. On CPU, when ``flashspec-asm-kernel`` is installed, the ABI
matches, and the processor supports AVX2, the residual is evaluated by
that assembly kernel. Every other case uses the pure-PyTorch reference
below. The Triton GPU acceptance test is unchanged.

The acceptance criterion and residual distribution are immutable by AGENTS.md §2.1.
Do NOT alter the residual formula without a corresponding paper revision.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch

from flashspec.utils.logging import get_logger

__all__ = [
    "ASM_NAN_CLAMP_TO_ZERO",
    "active_residual_backend",
    "rejection_sample",
    "residual_backend_status",
    "residual_distribution",
]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Numerical guard constants
# ---------------------------------------------------------------------------
_MIN_PROB: float = 1e-9  # Denominator guard for residual normalisation.

# NaN contract (flashspec-asm ADR 0001):
# ``torch.clamp(NaN, min=0)`` preserves NaN. AVX2 ``vmaxps(NaN, 0)`` returns
# the second operand, so the assembly kernel writes 0. Finite inputs are
# required to match within 1e-5. This flag documents the divergence; it does
# not change PyTorch behaviour.
ASM_NAN_CLAMP_TO_ZERO: bool = True

# ---------------------------------------------------------------------------
# Optional ASM accelerator backend
#
# flashspec-asm-kernel ships as a separate package:
#   pip install flashspec[asm]     # Linux x86-64; installs flashspec-asm-kernel
#   pip install flashspec          # pure-PyTorch residual, unchanged
#
# The kernel accelerates residual_dist on the CPU path only.
# ---------------------------------------------------------------------------
_EXPECTED_ASM_ABI: int = 1
_ASM_AVAILABLE: bool = False
_asm_residual_dist: Callable[..., Any] | None = None
_ASM_SKIP_REASON: str = "backend not initialised"


def _resolve_asm_backend() -> tuple[bool, Callable[..., Any] | None, str]:
    """Load the optional assembly residual kernel, or explain the fallback.

    The kernel is selected only when the import succeeds, ``abi_version()``
    equals :data:`_EXPECTED_ASM_ABI`, and ``cpu_supports_avx2()`` is true.
    Any failure selects pure PyTorch. A wrong ABI never runs the binary.

    Returns
    -------
    tuple[bool, Callable | None, str]
        ``(True, residual_dist, "selected")`` when the kernel should be
        used. Otherwise ``(False, None, reason)``.
    """
    try:
        import flashspec_asm_kernel as kernel
    except (ImportError, OSError) as exc:
        return (
            False,
            None,
            "flashspec-asm-kernel is not installed or its shared library "
            f"failed to load ({type(exc).__name__}: {exc})",
        )
    try:
        abi = int(kernel.abi_version())
        has_avx2 = bool(kernel.cpu_supports_avx2())
        residual_fn = kernel.residual_dist
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
        return (
            False,
            None,
            f"flashspec-asm-kernel failed during ABI or CPUID checks ({type(exc).__name__}: {exc})",
        )
    if abi != _EXPECTED_ASM_ABI:
        warnings.warn(
            "flashspec-asm-kernel ABI mismatch: expected "
            f"{_EXPECTED_ASM_ABI}, got {abi}. Falling back to the "
            "pure-PyTorch residual distribution.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False, None, f"ABI mismatch: expected {_EXPECTED_ASM_ABI}, got {abi}"
    if not has_avx2:
        return False, None, "cpu_supports_avx2() returned false"
    if not callable(residual_fn):
        return False, None, "flashspec_asm_kernel.residual_dist is not callable"
    return True, residual_fn, "selected"


def _install_asm_backend() -> None:
    """Resolve the assembly backend and store the module-level selection."""
    global _ASM_AVAILABLE, _asm_residual_dist, _ASM_SKIP_REASON
    available, residual_fn, reason = _resolve_asm_backend()
    _ASM_AVAILABLE = available
    _asm_residual_dist = residual_fn
    _ASM_SKIP_REASON = reason
    if available:
        logger.info("CPU residual backend: asm (flashspec-asm-kernel)")
    else:
        logger.info(
            "CPU residual backend: pytorch",
            extra={"reason": reason},
        )


_install_asm_backend()


def active_residual_backend() -> str:
    """Return the residual implementation currently selected.

    Returns
    -------
    str
        ``"asm"`` when the AVX2 kernel was selected at import, otherwise
        ``"pytorch"``. A CPU that has AVX2 but no loaded Linux shared
        library still returns ``"pytorch"``.
    """
    return "asm" if _ASM_AVAILABLE else "pytorch"


def residual_backend_status() -> dict[str, object]:
    """Describe the residual backend selection for logs and benchmarks.

    Returns
    -------
    dict[str, object]
        ``backend`` is ``"asm"`` or ``"pytorch"``. ``reason`` is ``"selected"``
        or a short explanation of the fallback. ``expected_abi`` is the ABI
        this package accepts.
    """
    from flashspec.utils.cpu import cpu_has_avx2

    return {
        "backend": active_residual_backend(),
        "reason": _ASM_SKIP_REASON,
        "host_avx2": cpu_has_avx2(),
        "expected_abi": _EXPECTED_ASM_ABI,
    }


def _pytorch_residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute ``max(0, p - q) / sum`` with the PyTorch reference.

    Parameters
    ----------
    p : torch.Tensor
        Target probabilities. The last dimension is the vocabulary.
    q : torch.Tensor
        Draft probabilities, same shape as ``p``.

    Returns
    -------
    torch.Tensor
        Residual distribution, same shape as ``p``. NaN inputs stay NaN.
    """
    residual = torch.clamp(p - q, min=0.0)
    denom = residual.sum(dim=-1, keepdim=True).clamp(min=_MIN_PROB)
    return residual / denom


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute the Algorithm 1 residual distribution.

    The formula is ``max(0, p - q) / sum(max(0, p - q))`` along the last
    dimension, with a ``1e-9`` denominator floor. Rows are sent to the
    assembly kernel only when all of the following hold: the kernel was
    selected, both tensors are contiguous CPU ``float32``, and the shape
    is ``(V,)`` or ``(batch, V)``. Otherwise the pure-PyTorch reference
    runs, including for CUDA tensors.

    Parameters
    ----------
    p : torch.Tensor
        Target probabilities.
    q : torch.Tensor
        Draft probabilities. Must have the same shape as ``p``.

    Returns
    -------
    torch.Tensor
        Residual probabilities, same shape as ``p``.

    Raises
    ------
    ValueError
        If ``p`` and ``q`` have different shapes.

    Notes
    -----
    On finite inputs the assembly kernel matches this function within
    absolute tolerance ``1e-5``. On NaN inputs the backends differ: PyTorch
    preserves NaN, and AVX2 ``vmaxps`` writes 0. See ``ASM_NAN_CLAMP_TO_ZERO``.
    """
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape; got {tuple(p.shape)} vs {tuple(q.shape)}."
        )
    use_asm = (
        _ASM_AVAILABLE
        and _asm_residual_dist is not None
        and p.device.type == "cpu"
        and p.dtype == torch.float32
        and q.dtype == torch.float32
        and p.is_contiguous()
        and q.is_contiguous()
        and p.ndim in (1, 2)
    )
    if not use_asm:
        return _pytorch_residual(p, q)

    squeeze = p.ndim == 1
    p_rows = p.unsqueeze(0) if squeeze else p
    q_rows = q.unsqueeze(0) if squeeze else q
    out = torch.empty_like(p_rows)
    residual_fn = _asm_residual_dist
    assert residual_fn is not None
    try:
        for row in range(p_rows.shape[0]):
            residual_fn(
                p_rows[row].detach().numpy(),
                q_rows[row].detach().numpy(),
                out[row].numpy(),
            )
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        logger.warning(
            "ASM residual kernel failed; using PyTorch for this call",
            extra={"error": str(exc)},
        )
        return _pytorch_residual(p, q)
    if squeeze:
        return out.squeeze(0)
    return out


def rejection_sample(
    input_ids: torch.Tensor,
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    gamma: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Speculative sampling via token-level acceptance/rejection (Algorithm 1).

    Implements the exact procedure from Leviathan et al. (2023).  The output
    distribution is identical to autoregressive sampling from the target model
    on finite probabilities. The CPU residual executor does not change the
    formula; see :func:`residual_distribution` for the NaN exception.

    Parameters
    ----------
    input_ids : torch.Tensor
        Current context token IDs.
        Shape: ``(batch_size, seq_len)``, dtype int64.
    draft_logprobs : torch.Tensor
        Log-probabilities under the draft model for each draft step.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
        Temperature **must** be applied to the raw logits by the caller before
        computing log-softmax (§7).  This function operates on the already-scaled
        log-probs and does not apply any further temperature rescaling.
    target_logprobs : torch.Tensor
        Log-probabilities under the target model at each draft position.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
        Temperature **must** be applied before log-softmax by the caller (§7).
    draft_token_ids : torch.Tensor
        Token indices proposed by the draft model.
        Shape: ``(batch_size, gamma)``, dtype int64.
    gamma : int
        Speculation length.  Must equal ``draft_logprobs.shape[1]``.

    Returns
    -------
    accepted_ids : torch.Tensor
        Accepted token IDs (draft tokens that passed the test, plus the
        residual token).  Shape: ``(batch_size, gamma)`` with ``-1`` padding
        at positions beyond ``first_rejection``.
    first_rejection : torch.Tensor
        Position of first rejection per sequence (or ``gamma`` if all accepted).
        Shape: ``(batch_size,)``, dtype int32.
    acceptance_rate : float
        Mean fraction of draft tokens accepted across the batch.

    Raises
    ------
    ValueError
        If tensor shapes are inconsistent or ``gamma`` does not match.
    RuntimeError
        If tensors are not on the same device.

    Notes
    -----
    When a draft token is rejected, its replacement is drawn from
    ``max(0, p - q) / sum(max(0, p - q))``. No temperature rescaling
    is applied to that residual, and it is not passed through softmax.

    Temperature scaling must be applied **before** log-softmax in the caller
    (``TargetModel.score_draft`` and the draft model's ``generate_draft``).
    See §7 of AGENTS.md.

    References
    ----------
    .. [1] Leviathan et al. (2023), "Fast Inference from Transformers via
       Speculative Decoding", arXiv:2211.17192, Algorithm 1.

    Examples
    --------
    >>> accepted_ids, first_rej, alpha = rejection_sample(
    ...     input_ids, draft_logprobs, target_logprobs, draft_token_ids, gamma=4
    ... )
    """
    if gamma != draft_logprobs.shape[1]:
        raise ValueError(
            f"gamma={gamma} does not match draft_logprobs.shape[1]={draft_logprobs.shape[1]}."
        )
    if draft_logprobs.shape != target_logprobs.shape:
        raise ValueError(
            f"draft_logprobs and target_logprobs must have identical shapes; "
            f"got {draft_logprobs.shape} vs {target_logprobs.shape}."
        )
    if draft_logprobs.device != target_logprobs.device:
        raise RuntimeError(
            f"All tensors must be on the same device; "
            f"got {draft_logprobs.device} vs {target_logprobs.device}."
        )

    batch_size, _gamma, vocab_size = draft_logprobs.shape
    device = draft_logprobs.device

    # Work in float32 for numerical stability.
    dlp = draft_logprobs.float()
    tlp = target_logprobs.float()

    # Gather log-probs at the draft token positions.
    ids = draft_token_ids.to(device).unsqueeze(-1)  # (B, γ, 1)
    lp_q = dlp.gather(-1, ids).squeeze(-1)  # (B, γ)
    lp_p = tlp.gather(-1, ids).squeeze(-1)  # (B, γ)

    # Sample uniform variates.
    u = torch.rand_like(lp_q)  # (B, γ)

    # Acceptance test.
    log_ratio = lp_p - lp_q
    accept_prob = log_ratio.exp().clamp(max=1.0)
    accepted_mask = u < accept_prob  # (B, γ) bool

    # first_rejection per sequence.
    rejected = ~accepted_mask
    has_rejection = rejected.any(dim=-1)
    first_rej_raw = rejected.int().argmax(dim=-1)
    first_rejection = torch.where(
        has_rejection,
        first_rej_raw,
        torch.full_like(first_rej_raw, gamma),
    ).to(torch.int32)

    # Acceptance rate for metrics.
    acceptance_rate = float(accepted_mask.float().mean().item())

    # Build accepted_ids: accepted draft tokens + residual token.
    # For each sequence, tokens 0..first_rejection-1 are accepted,
    # then one residual token is sampled from the adjusted distribution.
    accepted_ids = draft_token_ids.to(device).clone()
    # Tokens after the first rejection are discarded, even if that later
    # draft token would have passed its own test. Speculative decoding
    # stops at the first rejection.
    positions = torch.arange(gamma, device=device).unsqueeze(0)
    rejection_index = first_rejection.to(dtype=torch.long)
    after_rejection = positions > rejection_index.unsqueeze(1)
    accepted_ids = accepted_ids.masked_fill(~accepted_mask | after_rejection, -1)

    # A rejected row gets one residual token in the rejection slot.
    # A row that accepted every draft token keeps those tokens. Sampling
    # a residual there is undefined when p == q (the mass is zero) and
    # would replace an already-accepted token.
    rejected_rows = rejection_index < gamma
    if bool(rejected_rows.any().item()):
        residual_token_ids = _sample_residual(
            tlp, dlp, first_rejection, batch_size, gamma, vocab_size, device
        )
        row_index = torch.arange(batch_size, device=device)[rejected_rows]
        accepted_ids[row_index, rejection_index[rejected_rows]] = residual_token_ids[
            rejected_rows
        ].to(accepted_ids.dtype)

    logger.debug(
        "rejection_sample complete",
        extra={"acceptance_rate": acceptance_rate, "batch_size": batch_size, "gamma": gamma},
    )
    return accepted_ids, first_rejection, acceptance_rate


def _sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    first_rejection: torch.Tensor,
    batch_size: int,
    gamma: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample the residual token at the first rejection position.

    Parameters
    ----------
    target_probs : torch.Tensor
        Target model log-probabilities (float32).
        Shape: ``(batch_size, gamma, vocab_size)``.
    draft_probs : torch.Tensor
        Draft model log-probabilities (float32).
        Shape: ``(batch_size, gamma, vocab_size)``.
    first_rejection : torch.Tensor
        Index of first rejection per sequence.  Shape: ``(batch_size,)``, int32.
    batch_size : int
        Batch dimension.
    gamma : int
        Speculation length.
    vocab_size : int
        Vocabulary size.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Residual token IDs.  Shape: ``(batch_size,)``, dtype int64.
    """
    # The parameter names are historical: callers pass log-probabilities.
    p = target_probs.exp()  # (B, γ, V)
    q = draft_probs.exp()  # (B, γ, V)

    # Gather the (V,) slice at first_rejection for each batch element.
    safe_pos = first_rejection.long().clamp(max=gamma - 1)  # (B,)
    idx = safe_pos[:, None, None].expand(batch_size, 1, vocab_size)  # (B, 1, V)
    p_at_rej = p.gather(1, idx).squeeze(1)  # (B, V)
    q_at_rej = q.gather(1, idx).squeeze(1)  # (B, V)

    # Residual distribution: max(0, p-q) / sum(max(0, p-q)) — Algorithm 1.
    # CPU + AVX2 + a matching flashspec-asm-kernel build uses the assembly
    # kernel inside residual_distribution. GPU tensors and every other case
    # stay on the pure-PyTorch reference.
    residual = residual_distribution(p_at_rej, q_at_rej)
    # p == q makes the residual mass exactly 0. A rejection in that case is
    # numerical noise. Sample from the target distribution instead of handing
    # an all-zero vector to multinomial.
    mass = residual.sum(dim=-1, keepdim=True)
    usable = torch.isfinite(mass) & (mass > 1e-6)
    target = p_at_rej.clamp(min=0)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=_MIN_PROB)
    probs = torch.where(usable, residual, target)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=_MIN_PROB)

    # Sample from the residual distribution (always on torch).
    residual_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    if residual_token_ids.device != device:
        residual_token_ids = residual_token_ids.to(device)
    return residual_token_ids
