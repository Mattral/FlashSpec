"""Typical-acceptance sampling on a peaked toy distribution."""

from __future__ import annotations

import pytest
import torch

from flashspec.sampling.typical import typical_sample


def _peaked(
    token: int,
    gamma: int = 3,
    batch: int = 2,
    vocab: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.full((batch, gamma, vocab), -20.0)
    logits[..., token] = 20.0
    target = logits.log_softmax(-1)
    return target, target.clone()


def test_accepts_the_mode() -> None:
    """A draft of the target mode stays inside the typical set."""
    target, draft = _peaked(3)
    ids = torch.full((2, 3), 3)
    accepted, first_rejection, rate = typical_sample(draft, target, ids, gamma=3, typical_p=0.9)
    assert torch.equal(accepted, ids)
    assert torch.equal(first_rejection, torch.full((2,), 3, dtype=torch.int32))
    assert rate == 1.0


def test_rejects_an_unlikely_token() -> None:
    """A token outside the peak is rejected at the first position; the target fills it."""
    target, draft = _peaked(3)
    ids = torch.zeros((2, 3), dtype=torch.long)
    accepted, first_rejection, rate = typical_sample(draft, target, ids, gamma=3, typical_p=0.9)
    # Position 0 holds a draw from the (peaked) target; nothing after it.
    assert torch.equal(accepted, torch.tensor([[3, -1, -1], [3, -1, -1]]))
    assert torch.equal(first_rejection, torch.zeros(2, dtype=torch.int32))
    assert rate == 0.0


def test_only_the_prefix_before_the_first_rejection_is_kept() -> None:
    """A typical token after a rejected one is dropped; the rejection is filled from the target."""
    target, draft = _peaked(3, gamma=4, batch=1)
    ids = torch.tensor([[3, 0, 3, 3]])  # position 1 is rejected, 2 and 3 are typical
    accepted, first_rejection, rate = typical_sample(draft, target, ids, gamma=4, typical_p=0.9)
    assert torch.equal(accepted, torch.tensor([[3, 3, -1, -1]]))
    assert torch.equal(first_rejection, torch.ones(1, dtype=torch.int32))
    assert rate == 0.25


def test_the_draw_at_the_rejection_follows_the_target_and_the_generator() -> None:
    """The filled token is sampled from p at that position, reproducibly with a generator."""
    vocab = 6
    target = torch.log_softmax(torch.tensor([[[0.0, 1.0, 2.0, 0.5, -1.0, 1.5]]]), dim=-1)
    ids = torch.tensor([[4]])  # far below the entropy: rejected
    draws = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(4000):
        accepted, first_rejection, _ = typical_sample(
            target, target, ids, gamma=1, typical_p=0.9, generator=generator
        )
        assert int(first_rejection) == 0
        draws.append(int(accepted[0, 0]))
    counts = torch.bincount(torch.tensor(draws), minlength=vocab).float()
    expected = target[0, 0].exp() * len(draws)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 15.086, chi2  # chi-square 1% point, df 5
    again = typical_sample(
        target, target, ids, gamma=1, typical_p=0.9, generator=torch.Generator().manual_seed(0)
    )[0]
    first = typical_sample(
        target, target, ids, gamma=1, typical_p=0.9, generator=torch.Generator().manual_seed(0)
    )[0]
    assert torch.equal(again, first)


def test_rejects_bad_arguments() -> None:
    """typical_p and the log-prob shapes are checked before sampling."""
    target, draft = _peaked(1, gamma=2, batch=1)
    ids = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="typical_p"):
        typical_sample(draft, target, ids, gamma=2, typical_p=0.0)
    with pytest.raises(ValueError, match="typical_p"):
        typical_sample(draft, target, ids, gamma=2, typical_p=1.1)
    with pytest.raises(ValueError, match="identical shapes"):
        typical_sample(draft[..., :4], target, ids, gamma=2)
