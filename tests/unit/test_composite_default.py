"""Composite drafter strategies: separate gammas, the config setting, and the engine.

Proposal (a): separate lookup and fallback gammas on a composite arm, the
``drafter_strategy`` setting, and the engine building its arms from it.
Toy models on CPU: correctness only, no speed.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Iterator

import pytest
import torch
from pydantic import ValidationError
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import (
    COMPOSITE_STRATEGIES,
    DRAFTER_STRATEGIES,
    ArmSpec,
    strategy_arms,
    validate_arms,
)
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import DRAFTER_PAD_WIDTHS, SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import count_host_syncs
from flashspec.utils.config import (
    BanditConfig,
    DrafterStrategy,
    FlashSpecConfig,
    SamplingConfig,
)

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


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(21)
    out = [torch.randint(0, _VOCAB, (1, 7), generator=generator)]
    for length in (6, 12):
        chunk = torch.randint(0, _VOCAB, (1, length), generator=generator)
        out.append(torch.cat([chunk, chunk[:, : length // 2]], dim=1))
    return out


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _strategy_engine(
    target_model: LlamaForCausalLM,
    strategy: DrafterStrategy,
    *,
    static: bool,
    temperature: float = 0.0,
) -> SpeculativeEngine:
    """Engine whose arms come from ``drafter_strategy`` (``arms=None``)."""
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
    drafter = HuggingFaceDrafter("d", device="cpu", model=_perturbed(10, 0.003))
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=_NEW,
        drafter_strategy=strategy,
        sampling=SamplingConfig(gamma=3, temperature=temperature, seed=1),
    )
    engine = SpeculativeEngine(
        config,
        [drafter, NgramDrafter()],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=1, window_size=0),
    )
    if static:
        width = engine.arms[0].max_gamma + 1
        target.use_static_cache(64, compile_forward=False, pad_widths=(width,))
        drafter.use_static_cache(64, compile_forward=False, pad_widths=DRAFTER_PAD_WIDTHS)
    return engine


class TestArmSpecGammas:
    """``ArmSpec.fallback_gamma``, ``step_gamma`` and ``max_gamma``."""

    def test_step_and_max_gamma(self) -> None:
        """The fallback drafts ``fallback_gamma``; ``None`` keeps ``gamma``."""
        plain = ArmSpec(1, 4, fallback=0)
        split = ArmSpec(1, 6, fallback=0, fallback_gamma=2)
        wide_fallback = ArmSpec(1, 2, fallback=0, fallback_gamma=5)
        assert (plain.step_gamma(fallback=True), plain.step_gamma(fallback=False)) == (4, 4)
        assert (split.step_gamma(fallback=True), split.step_gamma(fallback=False)) == (2, 6)
        assert (plain.max_gamma, split.max_gamma, wide_fallback.max_gamma) == (4, 6, 5)
        assert ArmSpec(0, 3).max_gamma == 3

    def test_validation(self) -> None:
        """``fallback_gamma`` needs a fallback and must be >= 1."""
        lookup = [False, True]
        validate_arms([ArmSpec(1, 6, fallback=0, fallback_gamma=2)], 2, lookup)
        with pytest.raises(ValueError, match="no fallback"):
            validate_arms([ArmSpec(1, 6, fallback_gamma=2)], 2, lookup)
        with pytest.raises(ValueError, match="fallback_gamma 0"):
            validate_arms([ArmSpec(1, 6, fallback=0, fallback_gamma=0)], 2, lookup)


class TestStrategies:
    """Strategy names and the arms they build."""

    def test_names_match_the_config_literal(self) -> None:
        """``DrafterStrategy`` and ``DRAFTER_STRATEGIES`` list the same names."""
        assert set(typing.get_args(DrafterStrategy)) == set(DRAFTER_STRATEGIES)
        assert COMPOSITE_STRATEGIES == {"composite-g4": (4, 4), "composite-g6-f2": (6, 2)}

    def test_arms(self) -> None:
        """Bandit: one arm per drafter. Composite: one arm, first lookup over first model."""
        lookup = [False, True]
        assert strategy_arms("bandit", lookup, 3) == [ArmSpec(0, 3), ArmSpec(1, 3)]
        (g4,) = strategy_arms("composite-g4", lookup, 3)
        # The Phase 6 composite arm, by another label.
        assert (g4.drafter, g4.gamma, g4.fallback, g4.fallback_gamma) == (1, 4, 0, None)
        (g6,) = strategy_arms("composite-g6-f2", [True, False, False], 3)
        assert (g6.drafter, g6.gamma, g6.fallback, g6.fallback_gamma) == (0, 6, 1, 2)
        assert g6.label == "composite-g6-f2"
        with pytest.raises(ValueError, match="lookup drafter"):
            strategy_arms("composite-g4", [False, False], 3)
        with pytest.raises(ValueError, match="Unknown drafter strategy"):
            strategy_arms("nope", lookup, 3)


class TestConfig:
    """``FlashSpecConfig.drafter_strategy``."""

    def test_default_is_the_bandit(self) -> None:
        """Until the comparison run decides, the default stays 0.2.0's."""
        config = FlashSpecConfig()
        assert config.drafter_strategy == "bandit"
        assert config.bandit.n_arms == 2

    def test_composite_is_one_arm(self) -> None:
        """An omitted bandit gets ``n_arms=1``; an explicit other count is an error."""
        for strategy in COMPOSITE_STRATEGIES:
            config = FlashSpecConfig(drafter_strategy=strategy)  # type: ignore[arg-type]
            assert config.bandit.n_arms == 1
        FlashSpecConfig(drafter_strategy="composite-g4", bandit=BanditConfig(n_arms=1))
        with pytest.raises(ValidationError, match="n_arms must be 1"):
            FlashSpecConfig(drafter_strategy="composite-g4", bandit=BanditConfig(n_arms=2))
        with pytest.raises(ValidationError):
            FlashSpecConfig(drafter_strategy="composite-g5")  # type: ignore[arg-type]


class TestEngine:
    """The engine with arms from a strategy."""

    def test_arms_come_from_the_strategy(self, caplog: pytest.LogCaptureFixture) -> None:
        """``arms=None`` builds the strategy's arm; no single-arm warning for it."""
        logging.getLogger("flashspec.engine.speculative").setLevel(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            engine = _strategy_engine(_llama(3, 10), "composite-g6-f2", static=False)
        assert engine.arms == (ArmSpec(1, 6, "composite-g6-f2", fallback=0, fallback_gamma=2),)
        assert "cannot adapt" not in caplog.text

    def test_composite_needs_both_drafters(self) -> None:
        """Without a lookup drafter the composite strategy is refused."""
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(2, 1))
        drafter = HuggingFaceDrafter("d", device="cpu", model=_llama(2, 2))
        config = FlashSpecConfig(device="cpu", dtype="float32", drafter_strategy="composite-g4")
        with pytest.raises(ValueError, match="lookup drafter"):
            SpeculativeEngine(config, [drafter], target, UCB1Selector(n_arms=1))

    @pytest.mark.parametrize("static", [False, True])
    def test_separate_gammas_are_exact_and_used(
        self, static: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """composite-g6-f2: greedy equals HF; fallback steps draft 2, lookup steps up to 6."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        target_model = _llama(3, 10)
        engine = _strategy_engine(target_model, "composite-g6-f2", static=static)
        used = set()
        for prompt in _prompts():
            result = engine.generate(prompt)
            assert torch.equal(result.output_ids, _hf_greedy(target_model, prompt))
            context = prompt[0].tolist()
            output = result.output_ids[0].tolist()
            for step in result.arm_steps:
                _ids, n_valid = NgramDrafter().propose(torch.tensor([context]), len(context), 6)
                if int(n_valid) > 0:
                    assert step.drafter == 1
                    assert step.proposed == int(n_valid) <= 6
                else:
                    assert step.drafter == 0
                    assert step.proposed == 2
                used.add(step.drafter)
                context += output[len(context) - prompt.shape[1] :][: step.emitted]
        assert used == {0, 1}
        if static:
            # One verify width: gamma_max + 1 = 7, for both kinds of step.
            shapes = engine._target._inc.graph_call_shapes  # type: ignore[attr-defined]
            assert {width for width, _rows in shapes} == {7}

    def test_composite_g4_strategy_is_the_phase6_arm(self) -> None:
        """The ``composite-g4`` strategy runs exactly like ``ArmSpec(1, 4, fallback=0)``."""
        target_model = _llama(3, 10)
        by_strategy = _strategy_engine(target_model, "composite-g4", static=False)
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
        drafter = HuggingFaceDrafter("d", device="cpu", model=_perturbed(10, 0.003))
        config = FlashSpecConfig(
            device="cpu",
            dtype="float32",
            max_new_tokens=_NEW,
            bandit=BanditConfig(n_arms=1, window_size=0),
            sampling=SamplingConfig(gamma=4, temperature=0.0, seed=1),
        )
        explicit = SpeculativeEngine(
            config,
            [drafter, NgramDrafter()],  # type: ignore[list-item]
            target,
            UCB1Selector(n_arms=1, window_size=0),
            arms=[ArmSpec(1, 4, label="composite-g4", fallback=0)],
        )
        for prompt in _prompts():
            a, b = by_strategy.generate(prompt), explicit.generate(prompt)
            assert torch.equal(a.output_ids, b.output_ids)
            assert [(s.drafter, s.proposed, s.accepted) for s in a.arm_steps] == [
                (s.drafter, s.proposed, s.accepted) for s in b.arm_steps
            ]

    @pytest.mark.parametrize("temperature", [0.0, 0.8])
    def test_one_read_per_step_plus_one(self, temperature: float) -> None:
        """Separate gammas keep the composite's one device read per step (+1 per call)."""
        engine = _strategy_engine(
            _llama(3, 10), "composite-g6-f2", static=False, temperature=temperature
        )
        with count_host_syncs("cpu") as counter:
            result = engine.generate(_prompts()[1])
        assert counter.count == len(result.arm_steps) + 1, counter.by_kind
        assert result.n_tokens_generated == _NEW
