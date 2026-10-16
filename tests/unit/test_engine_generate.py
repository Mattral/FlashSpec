"""SpeculativeEngine generation with more than one drafter."""

from __future__ import annotations

import logging

import pytest
import torch

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig


class _Peaked:
    """Drafter or target that puts all mass on one token."""

    def __init__(self, vocab: int, peak: int) -> None:
        self.vocab = vocab
        self.peak = peak

    def _logprobs(self, batch: int, steps: int) -> torch.Tensor:
        logits = torch.full((batch, steps, self.vocab), -6.0)
        logits[:, :, self.peak] = 6.0
        return torch.log_softmax(logits, dim=-1)

    def generate_draft(
        self, input_ids: torch.Tensor, gamma: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose the peak token ``gamma`` times."""
        batch = int(input_ids.shape[0])
        ids = torch.full((batch, gamma), self.peak, dtype=torch.long)
        return ids, self._logprobs(batch, gamma)

    def compute_logprobs(
        self, input_ids: torch.Tensor, draft_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Score under the same peak."""
        return self._logprobs(int(draft_token_ids.shape[0]), int(draft_token_ids.shape[1]))

    def score_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        gamma: int,
        temperature: float = 1.0,
        include_bonus: bool = False,
    ) -> torch.Tensor:
        """Return verifier log-probs. Temperature is applied before log-softmax."""
        steps = gamma + 1 if include_bonus else gamma
        logits = torch.full((input_ids.shape[0], steps, self.vocab), -6.0)
        logits[:, :, self.peak] = 6.0
        if temperature != 1.0:
            logits = logits / temperature
        return torch.log_softmax(logits, dim=-1)


def _engine(n_arms: int, peaks: list[int]) -> SpeculativeEngine:
    """Build an engine whose drafters and target share a vocabulary of 16."""
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=6,
        bandit=BanditConfig(n_arms=n_arms, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=4, temperature=1.0),
    )
    drafters = [_Peaked(16, peak) for peak in peaks]
    target = _Peaked(16, peak=peaks[0])
    return SpeculativeEngine(
        config,
        drafters,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        UCB1Selector(n_arms=n_arms, window_size=0),
    )


class TestGenerate:
    """The loop returns a rectangular token tensor."""

    def test_single_arm_warns_and_generates(self, caplog: pytest.LogCaptureFixture) -> None:
        """One drafter still generates, and the warning says it cannot adapt."""
        with caplog.at_level(logging.WARNING):
            engine = _engine(1, [3])
        assert any("cannot adapt" in record.message for record in caplog.records)
        prompt = torch.randint(0, 16, (1, 4))
        result = engine.generate(prompt, max_new_tokens=6)
        assert result.output_ids.shape == (1, 6)
        assert result.n_tokens_generated == 6
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_two_drafters_batch_of_two(self) -> None:
        """Two equal-length prompts survive a two-arm engine."""
        engine = _engine(2, [3, 9])
        prompt = torch.randint(0, 16, (2, 4))
        result = engine.generate(prompt, max_new_tokens=4)
        assert result.output_ids.shape == (2, 4)
        assert int((result.output_ids >= 0).sum().item()) == 8

    def test_arm_count_mismatch_raises(self) -> None:
        """The engine refuses a drafter list that does not match n_arms."""
        config = FlashSpecConfig(
            device="cpu",
            dtype="float32",
            bandit=BanditConfig(n_arms=2),
            sampling=SamplingConfig(gamma=2),
        )
        with pytest.raises(ValueError, match="n_arms"):
            SpeculativeEngine(
                config,
                [_Peaked(8, 1)],  # type: ignore[list-item]
                _Peaked(8, 1),  # type: ignore[arg-type]
                UCB1Selector(n_arms=2),
            )
