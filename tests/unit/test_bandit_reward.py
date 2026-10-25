"""Bandit selection can use tokens per second without dropping the accept count."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flashspec.bandit.thompson import ThompsonSelector
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig


class _RecordingUCB(UCB1Selector):
    """UCB1 that stores the accepted count and the optional reward."""

    def __init__(self) -> None:
        super().__init__(n_arms=1, window_size=0)
        self.calls: list[tuple[int, float | None]] = []

    def update(self, arm: int, accepted: int, reward: float | None = None) -> None:
        """Record one engine update, then apply the normal UCB1 update."""
        self.calls.append((accepted, reward))
        super().update(arm, accepted, reward=reward)


class _Peaked:
    """Drafter and target that agree on one token, so the draft is accepted."""

    def __init__(self, vocab: int, peak: int) -> None:
        self.vocab = vocab
        self.peak = peak

    def generate_draft(
        self, input_ids: torch.Tensor, gamma: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose the peak token ``gamma`` times."""
        batch = int(input_ids.shape[0])
        ids = torch.full((batch, gamma), self.peak, dtype=torch.long)
        logits = torch.full((batch, gamma, self.vocab), -6.0)
        logits[:, :, self.peak] = 6.0
        return ids, torch.log_softmax(logits, dim=-1)

    def score_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        gamma: int,
        temperature: float = 1.0,
        include_bonus: bool = False,
    ) -> torch.Tensor:
        """Score the same peak, including the bonus row when asked."""
        del draft_token_ids, temperature
        steps = gamma + 1 if include_bonus else gamma
        logits = torch.full((int(input_ids.shape[0]), steps, self.vocab), -6.0)
        logits[:, :, self.peak] = 6.0
        return torch.log_softmax(logits, dim=-1)


def test_ucb_prefers_tokens_per_second_and_keeps_the_accept_count() -> None:
    """A faster arm with fewer accepts outranks a slow arm with more accepts."""
    selector = UCB1Selector(n_arms=2, window_size=0)
    assert selector.select() == 0
    selector.update(0, accepted=4, reward=4.0)
    assert selector.select() == 1
    selector.update(1, accepted=1, reward=10.0)
    assert selector.select() == 1
    assert selector._arms[0].n_accepted == 4
    assert selector._arms[1].n_accepted == 1
    assert selector._arms[0].mean_accept_rate == 4.0
    assert selector._arms[1].mean_reward == 10.0


def test_thompson_keeps_beta_on_the_accept_count() -> None:
    """A tokens-per-second reward does not enter the Beta posterior."""
    selector = ThompsonSelector(n_arms=1, window_size=0, prior_alpha=1.0, prior_beta=1.0)
    selector.update(0, accepted=1, reward=100.0)
    assert selector._post_alpha[0] == 2.0
    assert selector._post_beta[0] == 1.0
    assert selector._arms[0].n_accepted == 1
    assert selector._arms[0].mean_reward == 100.0


def test_omitted_reward_still_uses_the_accept_count() -> None:
    """Callers that pass only ``accepted`` keep the old UCB mean."""
    selector = UCB1Selector(n_arms=1, window_size=0)
    selector.update(0, accepted=1)
    assert selector._arms[0].n_accepted == 1
    assert selector._arms[0].mean_reward == 1.0
    assert selector._arms[0].mean_accept_rate == 1.0


def test_engine_rewards_emitted_tokens_per_step_second(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reward is emitted tokens (drafts + bonus) per step-second; the accept count is not."""
    import flashspec.engine.speculative as speculative

    ticks = iter(float(index) * 0.5 for index in range(1000))
    monkeypatch.setattr(speculative, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    bandit = _RecordingUCB()
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=5,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=4, temperature=1.0),
    )
    peaked = _Peaked(16, peak=3)
    engine = SpeculativeEngine(
        config,
        [peaked],  # type: ignore[list-item]
        peaked,  # type: ignore[arg-type]
        bandit,
    )
    result = engine.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=5)
    assert result.n_tokens_generated == 5
    assert len(bandit.calls) == 1
    accepted, reward = bandit.calls[0]
    # Four draft tokens accepted; the step emitted those plus the bonus.
    assert accepted == 4
    # The fake clock advances 0.5 s per read, and the step spans one read.
    assert reward == pytest.approx(5 / 0.5)
