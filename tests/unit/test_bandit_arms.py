"""Phase 6d: composite arm, per-step arm trace, and bandits on a tokens-per-second reward."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.thompson import ThompsonSelector
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import ArmSpec, validate_arms
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import DRAFTER_PAD_WIDTHS, SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import count_host_syncs
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

_VOCAB = 256
_NEW = 40


def _llama(layers: int, seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        initializer_range=0.1,
    )
    return LlamaForCausalLM(config).eval()


def _perturbed(seed: int, scale: float) -> LlamaForCausalLM:
    model = _llama(3, seed)
    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)
    return model


def _hf_greedy(model: LlamaForCausalLM, prompt: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=_NEW,
            min_new_tokens=_NEW,
            eos_token_id=None,
            pad_token_id=0,
        )
    return out[:, prompt.shape[1] :]


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _composite_engine(
    target_model: LlamaForCausalLM,
    *,
    static: bool,
    temperature: float = 0.0,
    extra_arms: bool = False,
) -> SpeculativeEngine:
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
    drafter = HuggingFaceDrafter("d", device="cpu", model=_perturbed(10, 0.003))
    if static:
        target.use_static_cache(64, compile_forward=False, pad_widths=(7,))
        drafter.use_static_cache(64, compile_forward=False, pad_widths=DRAFTER_PAD_WIDTHS)
    arms = [ArmSpec(1, 4, label="composite", fallback=0)]
    if extra_arms:
        arms += [ArmSpec(0, 2), ArmSpec(1, 6)]
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=_NEW,
        bandit=BanditConfig(n_arms=len(arms), window_size=0),
        sampling=SamplingConfig(gamma=4, temperature=temperature, seed=1),
        metrics=MetricsConfig(profile_steps=True),
    )
    return SpeculativeEngine(
        config,
        [drafter, NgramDrafter()],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=len(arms), window_size=0),
        arms=arms,
    )


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(21)
    out = [torch.randint(0, _VOCAB, (1, 7), generator=generator)]  # no repeats (likely)
    for length in (6, 12):
        chunk = torch.randint(0, _VOCAB, (1, length), generator=generator)
        out.append(torch.cat([chunk, chunk[:, : length // 2]], dim=1))
    return out


class TestCompositeArm:
    """Lookup when it has a proposal, the model drafter otherwise."""

    def test_validation(self) -> None:
        """A composite arm needs a lookup drafter and a model fallback."""
        lookup = [False, True]
        validate_arms([ArmSpec(1, 4, fallback=0)], 2, lookup)
        with pytest.raises(ValueError, match="composite"):
            validate_arms([ArmSpec(0, 4, fallback=1)], 2, lookup)
        with pytest.raises(ValueError, match="composite"):
            validate_arms([ArmSpec(1, 4, fallback=0)], 2, None)
        with pytest.raises(ValueError, match="fallback"):
            validate_arms([ArmSpec(1, 4, fallback=5)], 2, lookup)

    @pytest.mark.parametrize("static", [False, True])
    def test_choice_follows_the_lookup_and_output_is_exact(
        self, static: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each step's drafter is lookup iff the context has a match; greedy stays exact."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        target_model = _llama(3, 10)
        engine = _composite_engine(target_model, static=static)
        used = set()
        for prompt in _prompts():
            result = engine.generate(prompt)
            assert torch.equal(result.output_ids, _hf_greedy(target_model, prompt))
            context = prompt[0].tolist()
            output = result.output_ids[0].tolist()
            for step in result.arm_steps:
                buf = torch.tensor([context])
                _ids, n_valid = NgramDrafter().propose(buf, len(context), 4)
                expect = 1 if int(n_valid) > 0 else 0
                assert step.drafter == expect, (context, step)
                if expect == 1:
                    assert step.proposed == int(n_valid)
                used.add(step.drafter)
                context += output[len(context) - prompt.shape[1] :][: step.emitted]
        assert used == {0, 1}

    @pytest.mark.parametrize("temperature", [0.0, 0.8])
    def test_one_read_per_step_plus_one_per_call(self, temperature: float) -> None:
        """The composite choice rides on the step's read; only the first step needs one more.

        Dynamic cache: on CPU the eager static-cache path inside transformers
        makes its own tensor reads, which the compiled GPU path does not.
        """
        engine = _composite_engine(_llama(3, 10), static=False, temperature=temperature)
        with count_host_syncs("cpu") as counter:
            result = engine.generate(_prompts()[1])
        assert counter.count == len(result.arm_steps) + 1, counter.by_kind

    def test_trace_is_consistent(self) -> None:
        """Emitted tokens add up; reward is emitted / seconds; proposed bounds accepted."""
        engine = _composite_engine(_llama(3, 10), static=False, extra_arms=True)
        result = engine.generate(_prompts()[2])
        steps = result.arm_steps
        assert sum(step.emitted for step in steps) == result.n_tokens_generated == _NEW
        for step in steps:
            assert 0 <= step.accepted <= step.proposed
            assert step.reward == pytest.approx(step.emitted / step.seconds)
            assert step.drafter_pending >= 0
        assert {step.arm for step in steps} == {0, 1, 2}


# ── Bandits on a tokens-per-second reward ────────────────────────────────────


def _simulate(
    selector: UCB1Selector | ThompsonSelector, means: list[float], scale: float
) -> list[int]:
    """Gaussian rewards (sd 25% of the mean) in units of ``scale``; return the arm sequence."""
    rng = np.random.default_rng(3)
    picks = []
    for _ in range(600):
        arm = selector.select()
        reward = max(0.0, rng.normal(means[arm], 0.25 * means[arm])) * scale
        selector.update(arm, accepted=1, reward=reward)
        picks.append(arm)
    return picks


class TestUcbNormalisation:
    """UCB1 divides mean rewards by the largest reward seen."""

    def test_scale_invariant(self) -> None:
        """Same draws in tokens/s and in thousands of tokens/s: same choices."""
        means = [40.0, 55.0, 50.0]
        small = _simulate(UCB1Selector(n_arms=3, window_size=0), means, 1.0)
        large = _simulate(UCB1Selector(n_arms=3, window_size=0), means, 1000.0)
        assert small == large

    def test_explores_and_finds_the_best_arm(self) -> None:
        """Normalised: the best arm wins most pulls and the others are still tried."""
        picks = _simulate(UCB1Selector(n_arms=3, window_size=0), [40.0, 55.0, 50.0], 1.0)
        counts = [picks.count(k) for k in range(3)]
        assert counts[1] == max(counts)
        assert min(counts) > 1

    def test_unnormalised_stops_exploring(self) -> None:
        """Without scaling, a bonus of about 1 next to rewards of 40-55 never matters."""
        picks = _simulate(
            UCB1Selector(n_arms=3, window_size=0, normalise_rewards=False),
            [40.0, 55.0, 50.0],
            1.0,
        )
        counts = [picks.count(k) for k in range(3)]
        # After one pull each, one arm takes nearly everything.
        assert max(counts) >= 590

    def test_state_round_trip(self) -> None:
        """Scale and flag survive JSON; old states restore unscaled."""
        selector = UCB1Selector(n_arms=2, window_size=0)
        selector.update(0, accepted=2, reward=30.0)
        restored = UCB1Selector.from_json(selector.to_json())
        assert isinstance(restored, UCB1Selector)
        assert restored._reward_scale == 30.0 and restored._normalise is True
        state = json.loads(selector.to_json())
        state.pop("normalise_rewards")
        state.pop("reward_scale")
        old = UCB1Selector._from_state_dict(state)
        assert old._normalise is False


class TestThompsonReward:
    """Gaussian posterior on the reward when rewards are given."""

    def test_finds_the_best_arm_and_is_seeded(self) -> None:
        """Seeded: reproducible; the best arm takes most pulls."""
        means = [40.0, 55.0, 50.0]
        first = _simulate(ThompsonSelector(n_arms=3, window_size=0, seed=7), means, 1.0)
        second = _simulate(ThompsonSelector(n_arms=3, window_size=0, seed=7), means, 1.0)
        assert first == second
        counts = [first.count(k) for k in range(3)]
        assert counts[1] == max(counts)

    def test_scale_invariant(self) -> None:
        """The reward posterior does not depend on the unit."""
        means = [40.0, 55.0, 50.0]
        small = _simulate(ThompsonSelector(n_arms=3, window_size=0, seed=2), means, 1.0)
        large = _simulate(ThompsonSelector(n_arms=3, window_size=0, seed=2), means, 1000.0)
        assert small == large

    def test_beta_stays_positive_and_reset(self) -> None:
        """Accept counts above 1 add successes only; reset restores priors and mode."""
        selector = ThompsonSelector(n_arms=2, window_size=0, seed=0)
        selector.update(0, accepted=4, reward=None)
        assert selector._post_beta[0] == 1.0 and selector._post_alpha[0] == 5.0
        assert selector.select() in (0, 1)
        selector.update(1, accepted=0, reward=12.0)
        assert selector._reward_mode
        selector.reset()
        assert not selector._reward_mode and selector._post_alpha == [1.0, 1.0]

    def test_state_round_trip(self) -> None:
        """Seed and reward mode survive JSON."""
        selector = ThompsonSelector(n_arms=2, window_size=10, seed=5)
        selector.update(0, accepted=1, reward=3.0)
        restored = ThompsonSelector.from_json(selector.to_json())
        assert isinstance(restored, ThompsonSelector)
        assert restored._reward_mode and restored._seed == 5
        assert restored._arms[0].reward_sq_total == 9.0

    def test_variance_helpers(self) -> None:
        """Sample variance with and without a window."""
        selector = ThompsonSelector(n_arms=1, window_size=0, seed=0)
        for value in (1.0, 2.0, 3.0):
            selector.update(0, accepted=1, reward=value)
        assert selector._arms[0].reward_variance == pytest.approx(1.0)
        windowed = ThompsonSelector(n_arms=1, window_size=2, seed=0)
        for value in (1.0, 2.0, 4.0):
            windowed.update(0, accepted=1, reward=value)
        assert windowed._arms[0].reward_count == 2
        assert windowed._arms[0].reward_variance == pytest.approx(2.0)
        assert math.isfinite(windowed._pooled_variance())
