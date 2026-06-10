"""Standard speculative-sampling acceptance/rejection (Algorithm 1).

This module implements **exactly** Algorithm 1 from Leviathan et al. (2023),
"Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192).

The acceptance criterion and residual distribution are immutable by AGENTS.md §2.1.
Do NOT alter the residual computation below without a corresponding paper revision.
"""

from __future__ import annotations

import torch

from flashspec.utils.logging import get_logger

__all__ = ["rejection_sample"]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Numerical guard constants
# ---------------------------------------------------------------------------
_MIN_PROB: float = 1e-9  # Denominator guard for residual normalisation.


def rejection_sample(
    input_ids: torch.Tensor,
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    gamma: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Speculative sampling via token-level acceptance/rejection (Algorithm 1).

    Implements the exact procedure from Leviathan et al. (2023).  The output
    distribution is identical to autoregressive sampling from the target model.

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
    Residual distribution (when a draft token is rejected at position i):

    .. math::

        p_{\\text{residual}} = \\frac{\\max(0,\\, p - q)}{\\sum \\max(0,\\, p - q)}

    This is computed exactly as::

        residual = torch.clamp(p - q, min=0.0)
        residual = residual / residual.sum(dim=-1, keepdim=True)

    No temperature rescaling is applied to the residual.  No ``softmax``
    is applied to the residual (it is already a proper probability vector).

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
    ids = draft_token_ids.to(device).unsqueeze(-1)          # (B, γ, 1)
    lp_q = dlp.gather(-1, ids).squeeze(-1)                  # (B, γ)
    lp_p = tlp.gather(-1, ids).squeeze(-1)                  # (B, γ)

    # Sample uniform variates.
    u = torch.rand_like(lp_q)                               # (B, γ)

    # Acceptance test.
    log_ratio = lp_p - lp_q
    accept_prob = log_ratio.exp().clamp(max=1.0)
    accepted_mask = u < accept_prob                         # (B, γ) bool

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
    accepted_ids = draft_token_ids.clone()
    accepted_ids = accepted_ids.masked_fill(
        ~accepted_mask,
        -1,
    )

    # Sample the residual token at the rejection position (or gamma position
    # if all accepted, for the bonus token).
    residual_token_ids = _sample_residual(
        tlp, dlp, first_rejection, batch_size, gamma, vocab_size, device
    )

    # Insert residual at first_rejection position (replacing the rejected token
    # or appending as a bonus).
    batch_indices = torch.arange(batch_size, device=device)
    safe_pos = first_rejection.long().clamp(max=gamma - 1)
    accepted_ids[batch_indices, safe_pos] = residual_token_ids

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
        Target model probabilities (float32).  Shape: ``(batch_size, gamma, vocab_size)``.
    draft_probs : torch.Tensor
        Draft model probabilities (float32).  Shape: ``(batch_size, gamma, vocab_size)``.
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
    # Convert log-probs to probs.
    p = target_probs.exp()  # (B, γ, V)
    q = draft_probs.exp()   # (B, γ, V)

    # Gather the slice at first_rejection (clamped to valid range).
    safe_pos = first_rejection.long().clamp(max=gamma - 1)                   # (B,)
    idx = safe_pos[:, None, None].expand(batch_size, 1, vocab_size)           # (B, 1, V)
    p_at_rej = p.gather(1, idx).squeeze(1)                                    # (B, V)
    q_at_rej = q.gather(1, idx).squeeze(1)                                    # (B, V)

    # Residual distribution: clamp(p - q, min=0) / sum(...) — Algorithm 1 exact.
    residual = torch.clamp(p_at_rej - q_at_rej, min=0.0)
    denom = residual.sum(dim=-1, keepdim=True).clamp(min=_MIN_PROB)
    residual = residual / denom                                                # (B, V)

    # Sample from the residual distribution.
    residual_token_ids = torch.multinomial(residual, num_samples=1).squeeze(-1)  # (B,)
    return residual_token_ids
