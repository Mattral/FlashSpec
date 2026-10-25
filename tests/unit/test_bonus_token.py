"""A fully accepted step emits the bonus token from the target.

``score_draft`` used to return ``gamma`` positions and drop the logit
that predicts the token after the last draft token. A full accept then
yielded ``gamma`` tokens. With the bonus, ``p == q``, gamma 4, and
``max_new_tokens`` 5 finish in one ``score_draft`` call and emit 5 tokens.
A rejection does not emit that extra token.
"""

from __future__ import annotations

import torch
from torch import nn

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig


class _Logits:
    """Object with the ``logits`` attribute ``TargetModel`` reads."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _TinyCausal(nn.Module):
    """One bias vector, repeated at every position. Enough to score a slice."""

    def __init__(self, vocab: int = 8) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, input_ids: torch.Tensor) -> _Logits:
        """Return logits shaped like the input sequence."""
        batch, length = input_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).contiguous()
        return _Logits(logits)


class _CountingTarget:
    """Draft and target that share one peak, and count verifier calls."""

    def __init__(self, vocab: int, peak: int) -> None:
        self.vocab = vocab
        self.peak = peak
        self.calls = 0

    def _logprobs(self, batch: int, steps: int) -> torch.Tensor:
        logits = torch.full((batch, steps, self.vocab), -8.0)
        logits[:, :, self.peak] = 8.0
        return torch.log_softmax(logits, dim=-1)

    def generate_draft(
        self, input_ids: torch.Tensor, gamma: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose the peak token at every draft position."""
        batch = int(input_ids.shape[0])
        ids = torch.full((batch, gamma), self.peak, dtype=torch.long)
        return ids, self._logprobs(batch, gamma)

    def score_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        gamma: int,
        temperature: float = 1.0,
        include_bonus: bool = False,
    ) -> torch.Tensor:
        """Return ``gamma`` rows, or ``gamma + 1`` when the bonus is requested."""
        self.calls += 1
        steps = gamma + 1 if include_bonus else gamma
        logits = torch.full((int(input_ids.shape[0]), steps, self.vocab), -8.0)
        logits[:, :, self.peak] = 8.0
        if temperature != 1.0:
            logits = logits / temperature
        return torch.log_softmax(logits, dim=-1)


def _engine(
    target: _CountingTarget, drafter: _CountingTarget, max_new_tokens: int
) -> SpeculativeEngine:
    """One-arm engine. Gamma is 4."""
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=max_new_tokens,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=4, temperature=1.0),
    )
    return SpeculativeEngine(
        config,
        [drafter],  # type: ignore[list-item]
        target,  # type: ignore[arg-type]
        UCB1Selector(n_arms=1, window_size=0),
    )


def test_full_accept_emits_bonus_in_one_call() -> None:
    """``p == q``, gamma 4, ``max_new_tokens`` 5 is one verifier call and 5 tokens."""
    model = _CountingTarget(vocab=8, peak=3)
    engine = _engine(model, model, max_new_tokens=5)
    prompt = torch.zeros(1, 4, dtype=torch.long)
    result = engine.generate(prompt, max_new_tokens=5)
    assert model.calls == 1
    assert result.n_tokens_generated == 5
    assert result.output_ids.shape == (1, 5)
    assert bool((result.output_ids == 3).all())


def test_full_accept_does_not_pass_max_new_tokens() -> None:
    """Gamma 4 and ``max_new_tokens`` 4 keep four tokens and do not add a fifth."""
    model = _CountingTarget(vocab=8, peak=3)
    engine = _engine(model, model, max_new_tokens=4)
    prompt = torch.zeros(1, 4, dtype=torch.long)
    result = engine.generate(prompt, max_new_tokens=4)
    assert model.calls == 1
    assert result.n_tokens_generated == 4
    assert bool((result.output_ids == 3).all())


def test_rejection_does_not_emit_bonus() -> None:
    """A rejected draft emits the residual only, so two tokens take two calls."""
    drafter = _CountingTarget(vocab=8, peak=0)
    target = _CountingTarget(vocab=8, peak=1)
    engine = _engine(target, drafter, max_new_tokens=2)
    prompt = torch.zeros(1, 4, dtype=torch.long)
    result = engine.generate(prompt, max_new_tokens=2)
    assert target.calls == 2
    assert result.n_tokens_generated == 2
    assert bool((result.output_ids == 1).all())


def test_score_draft_include_bonus_returns_gamma_plus_one() -> None:
    """The bonus row is the logit after the last draft token."""
    module = _TinyCausal(vocab=8)
    target = TargetModel("tiny", torch.device("cpu"), torch.float32, model=module)
    context = torch.zeros(2, 5, dtype=torch.long)
    draft = torch.zeros(2, 4, dtype=torch.long)
    without = target.score_draft(context, draft, gamma=4, include_bonus=False)
    with_bonus = target.score_draft(context, draft, gamma=4, include_bonus=True)
    assert without.shape == (2, 4, 8)
    assert with_bonus.shape == (2, 5, 8)
