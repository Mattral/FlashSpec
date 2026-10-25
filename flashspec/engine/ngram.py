"""Prompt-lookup (n-gram) drafter.

The drafter looks up the last ``n`` tokens of the context in the earlier
context and proposes the tokens that followed their most recent earlier
occurrence. It tries ``n = 3``, then ``2``, then ``1`` (by default), and
proposes nothing when none of them occurs earlier. It runs no model, so
its cost is a handful of small device kernels.

Everything runs on the device with fixed shapes: the search compares
every window of the context buffer with the key at once (``unfold`` and
a compare), and the context length may be a device tensor. The proposal
length is a device tensor too, so drafting never reads the device.

The proposal is always ``gamma`` tokens wide. Only the first
``n_valid`` are real; the engine treats the rest like padding: they are
forwarded through the target (fixed shape) and then cropped, and they are
never accepted. ``n_valid`` may be ``0``; the step is then a plain decode
step that emits one target token.

For sampling, the drafter's distribution is one-hot on each proposed
token. The engine verifies it with Algorithm 1 for that ``q``
(:func:`flashspec.sampling.rejection.rejection_step_onehot`): accept with
probability ``min(1, p(x))``, and on a rejection sample the residual
``max(0, p - onehot(x))`` renormalised, which is ``p`` with ``x`` zeroed.
"""

from __future__ import annotations

import torch

__all__ = ["NgramDrafter", "is_lookup_drafter"]


def is_lookup_drafter(drafter: object) -> bool:
    """Whether ``drafter`` proposes by lookup (``propose``) instead of a model."""
    return callable(getattr(drafter, "propose", None))


class NgramDrafter:
    """Prompt-lookup drafter: continue the most recent earlier match of the last n tokens.

    Parameters
    ----------
    max_ngram : int
        Longest key tried first. Default 3.
    min_ngram : int
        Shortest key tried last. Default 1.
    name : str
        Label used in logs and results.

    Raises
    ------
    ValueError
        If ``min_ngram < 1`` or ``max_ngram < min_ngram``.
    """

    def __init__(self, max_ngram: int = 3, min_ngram: int = 1, name: str = "ngram") -> None:
        if min_ngram < 1 or max_ngram < min_ngram:
            raise ValueError(
                f"need 1 <= min_ngram <= max_ngram; got min_ngram={min_ngram}, "
                f"max_ngram={max_ngram}."
            )
        self._max_ngram = int(max_ngram)
        self._min_ngram = int(min_ngram)
        self._name = name

    @property
    def model_name(self) -> str:
        """Label of this drafter."""
        return self._name

    @property
    def ngram_sizes(self) -> tuple[int, ...]:
        """Key lengths in the order they are tried."""
        return tuple(range(self._max_ngram, self._min_ngram - 1, -1))

    def propose(
        self,
        tokens: torch.Tensor,
        length: int | torch.Tensor,
        gamma: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose up to ``gamma`` tokens for the context ``tokens[:, :length]``.

        Parameters
        ----------
        tokens : torch.Tensor
            Context buffer, shape ``(1, capacity)``, int64. Positions at and
            after ``length`` are ignored, whatever they hold.
        length : int or torch.Tensor
            Context length. A 0-d (or one-element) int64 device tensor is
            used as is, so the call does not read the device.
        gamma : int
            Proposal width. Must be ``>= 1``.

        Returns
        -------
        draft_ids : torch.Tensor
            Shape ``(1, gamma)``, int64. Entries at and after ``n_valid``
            are filler (a token of the context) and are never accepted.
        n_valid : torch.Tensor
            Shape ``(1,)``, int64, in ``[0, gamma]``: how many leading
            entries are a real proposal.

        Raises
        ------
        ValueError
            If ``tokens`` is not ``(1, capacity)`` or ``gamma < 1``.
        """
        if tokens.ndim != 2 or int(tokens.shape[0]) != 1:
            raise ValueError(f"tokens must be (1, capacity); got {tuple(tokens.shape)}.")
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1; got {gamma}.")
        seq = tokens[0]
        device = seq.device
        capacity = int(seq.shape[0])
        if isinstance(length, torch.Tensor):
            ctx_len = length.reshape(()).to(device=device, dtype=torch.long)
        else:
            # torch.full is a fill kernel; torch.tensor would copy from the
            # host and wait for the device.
            ctx_len = torch.full((), int(length), dtype=torch.long, device=device)

        none = torch.full((), -1, dtype=torch.long, device=device)
        best_start = none
        best_n = torch.zeros((), dtype=torch.long, device=device)
        for n in self.ngram_sizes:
            if capacity < n + 1:
                continue
            windows = seq.unfold(0, n, 1)  # (capacity - n + 1, n)
            key_pos = (ctx_len - n + torch.arange(n, device=device)).clamp(min=0, max=capacity - 1)
            key = seq.index_select(0, key_pos)
            starts = torch.arange(windows.shape[0], device=device)
            # An earlier occurrence lies wholly inside the context and has at
            # least one continuation token in it: start + n < length. That
            # also excludes the key itself (start = length - n).
            hit = (windows == key).all(dim=-1) & (starts + n < ctx_len) & (ctx_len > n)
            latest = torch.where(hit, starts, none).max()
            take = (best_start < 0) & (latest >= 0)
            best_start = torch.where(take, latest, best_start)
            best_n = torch.where(take, torch.full_like(best_n, n), best_n)

        found = best_start >= 0
        cont = (best_start + best_n).clamp(min=0)
        n_valid = torch.where(found, (ctx_len - cont).clamp(min=0, max=gamma), 0)
        offsets = torch.arange(gamma, device=device)
        picked = seq.index_select(0, (cont + offsets).clamp(max=capacity - 1))
        # Filler: the last context token, so every id fed to the target is a
        # real vocabulary id even where the buffer holds stale values.
        filler = seq.index_select(0, (ctx_len - 1).clamp(min=0).view(1))
        draft = torch.where(offsets < n_valid, picked, filler)
        return draft.view(1, gamma), n_valid.view(1)
