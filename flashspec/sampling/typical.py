"""Typical-acceptance speculative sampling variant.

This module implements a variant of speculative sampling based on
*typical acceptance*, where the acceptance decision is based on whether
the token's probability under the target model lies within a typical set
defined by entropy proximity.

Reference: Cai et al. (2024), "Medusa: Simple LLM Inference Acceleration
Framework with Multiple Decoding Heads."
"""

from __future__ import annotations

import torch

from flashspec.utils.logging import get_logger

__all__ = ["typical_sample"]

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_TYPICAL_EPSILON: float = 0.001   # Minimum acceptance floor.
_ENTROPY_CLAMP_MIN: float = 1e-9  # Guard for log(0) in entropy computation.


def typical_sample(
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    gamma: int,
    typical_p: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Typical-acceptance speculative sampling.

    Accepts a draft token if its (negative) log-probability under the target
    model is within the typical set defined by the target entropy.

    Parameters
    ----------
    draft_logprobs : torch.Tensor
        Log-probabilities under the draft model.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
    target_logprobs : torch.Tensor
        Log-probabilities under the target model.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
    draft_token_ids : torch.Tensor
        Token indices proposed by the draft model.
        Shape: ``(batch_size, gamma)``, dtype int64.
    gamma : int
        Speculation length.
    typical_p : float
        Mass of the typical set.  Tokens outside the set are rejected.
        Must be in ``(0, 1]``.

    Returns
    -------
    accepted_ids : torch.Tensor
        Accepted draft token IDs.  Shape: ``(batch_size, gamma)`` with
        ``-1`` padding for rejected positions.
    first_rejection : torch.Tensor
        Index of first rejection per sequence.
        Shape: ``(batch_size,)``, dtype int32.
    acceptance_rate : float
        Mean fraction of draft tokens accepted across the batch.

    Raises
    ------
    ValueError
        If ``typical_p`` is not in ``(0, 1]`` or shapes are inconsistent.

    Notes
    -----
    The typical-acceptance criterion:

    .. math::

        \\text{accept}_i = \\left|{-\\log p(x_i)} - H(p)\\right| \\le
            -\\log(\\text{typical\\_p})

    where :math:`H(p)` is the entropy of the target distribution.

    References
    ----------
    .. [1] Cai et al. (2024), "Medusa: Simple LLM Inference Acceleration
       Framework with Multiple Decoding Heads", arXiv:2401.10774.

    Examples
    --------
    >>> accepted_ids, first_rej, alpha = typical_sample(
    ...     draft_logprobs, target_logprobs, draft_token_ids, gamma=4, typical_p=0.9
    ... )
    """
    if not (0.0 < typical_p <= 1.0):
        raise ValueError(f"typical_p must be in (0, 1]; got {typical_p}.")
    if draft_logprobs.shape != target_logprobs.shape:
        raise ValueError(
            f"draft_logprobs and target_logprobs must have identical shapes; "
            f"got {draft_logprobs.shape} vs {target_logprobs.shape}."
        )

    batch_size, _gamma, vocab_size = draft_logprobs.shape
    device = target_logprobs.device

    tlp = target_logprobs.float()    # (B, γ, V)
    tp = tlp.exp()                   # (B, γ, V)

    # Entropy H(p) over vocab for each (batch, gamma) position.
    log_tp_safe = (tp + _ENTROPY_CLAMP_MIN).log()
    entropy = -(tp * log_tp_safe).sum(dim=-1)            # (B, γ)

    # Gather target log-prob at draft token.
    ids = draft_token_ids.to(device).unsqueeze(-1)       # (B, γ, 1)
    lp_x = tlp.gather(-1, ids).squeeze(-1)               # (B, γ)

    # Typical acceptance: |(-log p(x)) - H(p)| <= -log(typical_p)
    threshold = -torch.log(torch.tensor(typical_p, device=device, dtype=torch.float32))
    surprisal = -lp_x                                    # (B, γ)
    accepted_mask = (surprisal - entropy).abs() <= threshold   # (B, γ)

    # first_rejection
    rejected = ~accepted_mask
    has_rejection = rejected.any(dim=-1)
    first_rej_raw = rejected.int().argmax(dim=-1)
    first_rejection = torch.where(
        has_rejection,
        first_rej_raw,
        torch.full_like(first_rej_raw, gamma),
    ).to(torch.int32)

    accepted_ids = draft_token_ids.clone().masked_fill(~accepted_mask, -1)
    acceptance_rate = float(accepted_mask.float().mean().item())

    logger.debug(
        "typical_sample complete",
        extra={"acceptance_rate": acceptance_rate, "typical_p": typical_p},
    )
    return accepted_ids, first_rejection, acceptance_rate
