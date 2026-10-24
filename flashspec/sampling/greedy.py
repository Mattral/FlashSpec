"""Greedy verification (temperature 0).

With temperature 0 the target's output is its argmax at every position.
A draft token is accepted while it equals the target's argmax at that
position, and the step's next token is the target's argmax right after
the last accepted draft token. That one gather covers both cases: a
mismatch at ``i`` emits ``target_argmax[i]`` in place of the draft token,
and a full accept emits ``target_argmax[gamma]`` as the bonus token.

The output equals the target's own greedy decode whatever the drafter
proposed, so the drafter may propose by any rule. No softmax and no
randomness are involved.
"""

from __future__ import annotations

import torch

__all__ = ["greedy_verify"]


def greedy_verify(
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Accept draft tokens up to the first disagreement with the target argmax.

    Parameters
    ----------
    draft_token_ids : torch.Tensor
        Proposed tokens. Shape ``(batch, gamma)``, int64.
    target_argmax : torch.Tensor
        Target argmax at each draft position, plus optionally the bonus
        position. Shape ``(batch, gamma)`` or ``(batch, gamma + 1)``.

    Returns
    -------
    accepted_ids : torch.Tensor
        Shape ``(batch, gamma)``. Accepted draft tokens, then the target's
        token at the first mismatch, then ``-1`` padding. The same layout
        as :func:`flashspec.sampling.rejection.rejection_sample`.
    first_mismatch : torch.Tensor
        Accepted draft tokens per row, in ``[0, gamma]``. Shape
        ``(batch,)``, int32.
    bonus_ids : torch.Tensor or None
        ``target_argmax[:, gamma]``, the token after a full accept. Shape
        ``(batch,)``. ``None`` when ``target_argmax`` has only ``gamma``
        columns.

    Raises
    ------
    ValueError
        If the shapes do not line up.
    """
    if draft_token_ids.ndim != 2 or target_argmax.ndim != 2:
        raise ValueError("draft_token_ids and target_argmax must be 2-D.")
    batch, gamma = draft_token_ids.shape
    if target_argmax.shape[0] != batch or target_argmax.shape[1] not in (gamma, gamma + 1):
        raise ValueError(
            f"target_argmax must be ({batch}, {gamma}) or ({batch}, {gamma + 1}); "
            f"got {tuple(target_argmax.shape)}."
        )
    target_argmax = target_argmax.to(device=draft_token_ids.device, dtype=draft_token_ids.dtype)
    mismatch = draft_token_ids != target_argmax[:, :gamma]
    first = torch.where(
        mismatch.any(dim=-1),
        mismatch.int().argmax(dim=-1),
        torch.full((batch,), gamma, dtype=torch.long, device=draft_token_ids.device),
    )
    positions = torch.arange(gamma, device=draft_token_ids.device).unsqueeze(0)
    # Positions before the first mismatch keep the draft (it equals the
    # target there), the mismatch slot takes the target token, the rest pad.
    accepted = torch.where(positions <= first.unsqueeze(1), target_argmax[:, :gamma], -1)
    bonus = target_argmax[:, gamma] if target_argmax.shape[1] == gamma + 1 else None
    return accepted, first.to(torch.int32), bonus
