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
    """A token outside the peak is rejected at the first position."""
    target, draft = _peaked(3)
    ids = torch.zeros((2, 3), dtype=torch.long)
    accepted, first_rejection, rate = typical_sample(draft, target, ids, gamma=3, typical_p=0.9)
    assert torch.equal(accepted, torch.full_like(ids, -1))
    assert torch.equal(first_rejection, torch.zeros(2, dtype=torch.int32))
    assert rate == 0.0


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
