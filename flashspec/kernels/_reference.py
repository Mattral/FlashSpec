"""Pure-PyTorch reference implementations of FlashSpec kernels.

These implementations are the ground-truth against which every Triton kernel
is numerically validated.  They must NOT be used in production inference
paths — they exist solely for correctness testing.

See Also
--------
flashspec.kernels.verify_kernel : Triton-optimised equivalent of
    ``verify_tokens_reference``.
flashspec.kernels.gather_kernel : Triton-optimised equivalent of
    ``gather_accepted_reference``.
"""

from __future__ import annotations

import torch

__all__ = [
    "verify_tokens_reference",
    "gather_accepted_reference",
]


def verify_tokens_reference(
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    u: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference parallel token acceptance test (pure PyTorch).

    Implements the acceptance criterion from Algorithm 1 of Leviathan et al.
    (2023) in plain PyTorch.  The Triton kernel ``verify_tokens`` in
    ``flashspec.kernels.verify_kernel`` must match this output to within
    1e-5 (float32) / 1e-3 (bfloat16).

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
        Index of the first rejection per sequence, or ``gamma`` if all
        tokens were accepted.
        Shape: ``(batch_size,)``, dtype int32.

    Raises
    ------
    ValueError
        If ``draft_logprobs`` and ``target_logprobs`` shapes do not match,
        or if ``u`` has incorrect shape.
    RuntimeError
        If tensors are not on the same device.

    Notes
    -----
    The acceptance criterion is:

    .. math::

        \\text{accept}_i = u_i < \\min\\!\\left(1,\\,
            \\exp\\!\\left(\\log p(x_i) - \\log q(x_i)\\right)\\right)

    Arithmetic is performed in float32 to match Triton kernel precision.

    References
    ----------
    .. [1] Leviathan et al. (2023), "Fast Inference from Transformers via
       Speculative Decoding", arXiv:2211.17192, Algorithm 1.

    Examples
    --------
    >>> batch_size, gamma, vocab = 2, 4, 1000
    >>> dlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
    >>> tlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
    >>> u   = torch.rand(batch_size, gamma)
    >>> ids = torch.randint(0, vocab, (batch_size, gamma))
    >>> accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
    >>> accepted.shape
    torch.Size([2, 4])
    """
    if draft_logprobs.shape != target_logprobs.shape:
        raise ValueError(
            f"draft_logprobs and target_logprobs must have identical shapes; "
            f"got {draft_logprobs.shape} vs {target_logprobs.shape}."
        )
    batch_size, gamma, _vocab_size = draft_logprobs.shape
    if u.shape != (batch_size, gamma):
        raise ValueError(
            f"u must have shape (batch_size, gamma) = ({batch_size}, {gamma}); "
            f"got {u.shape}."
        )
    if draft_token_ids.shape != (batch_size, gamma):
        raise ValueError(
            f"draft_token_ids must have shape ({batch_size}, {gamma}); "
            f"got {draft_token_ids.shape}."
        )
    if draft_logprobs.device != target_logprobs.device:
        raise RuntimeError(
            f"draft_logprobs and target_logprobs must be on the same device; "
            f"got {draft_logprobs.device} vs {target_logprobs.device}."
        )

    # Upcast to float32 for numerical stability.
    dlp = draft_logprobs.float()
    tlp = target_logprobs.float()
    u_f32 = u.float().to(dlp.device)

    # Gather log-probs for the actual draft tokens.
    # ids: (batch, gamma) → (batch, gamma, 1) → gather → (batch, gamma, 1) → squeeze
    ids = draft_token_ids.to(dlp.device).unsqueeze(-1)  # (B, γ, 1)
    lp_q = dlp.gather(-1, ids).squeeze(-1)               # (B, γ)
    lp_p = tlp.gather(-1, ids).squeeze(-1)               # (B, γ)

    # Acceptance probability: min(1, p/q) = min(1, exp(log p - log q))
    log_ratio = lp_p - lp_q                              # (B, γ)
    accept_prob = log_ratio.exp().clamp(max=1.0)         # (B, γ)
    accepted = u_f32 < accept_prob                       # (B, γ)  bool

    # first_rejection: index of the first False in each row, or gamma if all True.
    # Using argmax on the inverted mask; handle the all-True case explicitly.
    rejected = ~accepted                                 # (B, γ)
    has_rejection = rejected.any(dim=-1)                 # (B,)
    first_rej_raw = rejected.int().argmax(dim=-1)        # (B,)  — 0 when no rejection
    first_rejection = torch.where(
        has_rejection, first_rej_raw, torch.full_like(first_rej_raw, gamma)
    )
    return accepted, first_rejection.to(torch.int32)


def gather_accepted_reference(
    token_ids: torch.Tensor,
    first_rejection: torch.Tensor,
) -> torch.Tensor:
    """Reference gather of accepted tokens up to the first rejection (pure PyTorch).

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
        Accepted token IDs packed with padding ``-1`` beyond ``first_rejection``.
        Shape: ``(batch_size, gamma)``, dtype int64.

    Raises
    ------
    ValueError
        If ``token_ids`` and ``first_rejection`` batch dimensions do not match.

    Notes
    -----
    This is the pure-PyTorch ground truth that the Triton ``gather_accepted``
    kernel in ``flashspec.kernels.gather_kernel`` must match exactly.
    Tests in ``tests/unit/test_verify_kernel.py`` compare both implementations.

    The masking uses a position broadcast:
    ``mask = positions < first_rejection`` where positions is
    ``arange(gamma).unsqueeze(0)``.

    Examples
    --------
    >>> ids = torch.tensor([[10, 20, 30, 40]])
    >>> frej = torch.tensor([2], dtype=torch.int32)
    >>> gather_accepted_reference(ids, frej)
    tensor([[10, 20, -1, -1]])
    """
    batch_size, gamma = token_ids.shape
    if first_rejection.shape != (batch_size,):
        raise ValueError(
            f"first_rejection must have shape ({batch_size},); "
            f"got {first_rejection.shape}."
        )
    positions = torch.arange(gamma, device=token_ids.device).unsqueeze(0)  # (1, γ)
    mask = positions < first_rejection.long().unsqueeze(1)                  # (B, γ)
    return token_ids.masked_fill(~mask, -1)
