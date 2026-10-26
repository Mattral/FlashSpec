"""SpeculativeEngine(..., bandit=None): optional with one arm, required with several.

Backward compatibility: passing a selector positionally or by keyword works as before.
Toy models on CPU: correctness only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import torch

from flashspec.bandit.base import DraftSelector
from flashspec.bandit.thompson import ThompsonSelector
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import ArmSpec
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig
from tests.unit.test_composite_default import _NEW, _hf_greedy, _llama, _perturbed, _prompts


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _config(**kwargs: object) -> FlashSpecConfig:
    return FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=_NEW,
        sampling=SamplingConfig(gamma=3, temperature=0.0),
        **kwargs,  # type: ignore[arg-type]
    )


def _target(model: torch.nn.Module) -> TargetModel:
    return TargetModel("t", torch.device("cpu"), torch.float32, model=model)


def _drafter() -> HuggingFaceDrafter:
    return HuggingFaceDrafter("d", device="cpu", model=_perturbed(10, 0.003))


class TestOneArmNeedsNoSelector:
    """The engine supplies a one-arm selector when the resolved strategy has one arm."""

    def test_auto_composite_matches_the_explicit_selector_and_hf(self) -> None:
        """Default config, one model drafter, no selector: composite-g6-f2, HF-exact greedy."""
        model = _llama(3, 10)
        implicit = SpeculativeEngine(_config(), [_drafter()], _target(model))
        explicit = SpeculativeEngine(
            _config(), [_drafter()], _target(model), UCB1Selector(n_arms=1)
        )
        assert implicit.drafter_strategy == explicit.drafter_strategy == "composite-g6-f2"
        assert isinstance(implicit._bandit, UCB1Selector) and implicit._bandit.n_arms == 1
        for prompt in _prompts():
            got = implicit.generate(prompt)
            assert torch.equal(got.output_ids, explicit.generate(prompt).output_ids)
            assert torch.equal(got.output_ids, _hf_greedy(model, prompt))

    def test_explicit_bandit_strategy_with_one_drafter(self) -> None:
        """drafter_strategy="bandit" and one drafter is one arm: no selector needed."""
        model = _llama(3, 10)
        engine = SpeculativeEngine(
            _config(drafter_strategy="bandit", bandit=BanditConfig(n_arms=1)),
            [_drafter()],
            _target(model),
        )
        assert engine.arms == (ArmSpec(0, 3),)
        prompt = _prompts()[1]
        assert torch.equal(engine.generate(prompt).output_ids, _hf_greedy(model, prompt))

    def test_explicit_single_arm(self) -> None:
        """arms=[one ArmSpec] with no selector."""
        engine = SpeculativeEngine(
            _config(bandit=BanditConfig(n_arms=1)),
            [_drafter(), NgramDrafter()],  # type: ignore[list-item]
            _target(_llama(3, 10)),
            arms=[ArmSpec(1, 4, fallback=0)],
        )
        assert engine.generate(_prompts()[2]).n_tokens_generated == _NEW

    def test_batch_fallback_with_no_selector(self) -> None:
        """Under auto, a batch-2 call falls back to the model drafter; still one arm, still fine."""
        model = _llama(3, 10)
        engine = SpeculativeEngine(_config(), [_drafter()], _target(model))
        prompt = torch.cat([_prompts()[0], _prompts()[0]], dim=0)
        result = engine.generate(prompt)
        assert result.n_tokens_generated == 2 * _NEW


class TestSeveralArmsNeedASelector:
    """With more than one arm, omitting the selector is an error that says what to pass."""

    def test_two_model_drafters(self) -> None:
        """Auto with two model drafters resolves to the bandit path: two arms."""
        with pytest.raises(
            ValueError, match=r"UCB1Selector\(n_arms=2\).*ThompsonSelector\(n_arms=2\)"
        ):
            SpeculativeEngine(_config(), [_drafter(), _drafter()], _target(_llama(3, 10)))

    def test_explicit_arms(self) -> None:
        """Three explicit arms and no selector."""
        with pytest.raises(ValueError, match=r"3 arms need a bandit selector"):
            SpeculativeEngine(
                _config(bandit=BanditConfig(n_arms=3)),
                [_drafter(), NgramDrafter()],  # type: ignore[list-item]
                _target(_llama(3, 10)),
                arms=[ArmSpec(0, 2), ArmSpec(1, 4), ArmSpec(1, 4, fallback=0)],
            )


class TestBackwardCompatible:
    """Existing call styles keep working unchanged."""

    @pytest.mark.parametrize(
        "selector", [UCB1Selector(n_arms=2, window_size=0), ThompsonSelector(n_arms=2, seed=0)]
    )
    def test_positional_and_keyword_selector(self, selector: DraftSelector) -> None:
        """A selector passed as the 4th positional argument or as bandit= is used as given."""
        config = _config(drafter_strategy="bandit", bandit=BanditConfig(n_arms=2))
        drafters = [_drafter(), _drafter()]
        positional = SpeculativeEngine(config, drafters, _target(_llama(3, 10)), selector)  # type: ignore[arg-type]
        keyword = SpeculativeEngine(config, drafters, _target(_llama(3, 10)), bandit=selector)  # type: ignore[arg-type]
        assert positional._bandit is selector and keyword._bandit is selector
        assert positional.generate(_prompts()[0]).n_tokens_generated == _NEW

    def test_arms_still_by_keyword(self) -> None:
        """arms= keeps its keyword, after the now-optional bandit."""
        engine = SpeculativeEngine(
            _config(bandit=BanditConfig(n_arms=1)),
            [_drafter()],
            _target(_llama(3, 10)),
            UCB1Selector(n_arms=1),
            arms=[ArmSpec(0, 2)],
        )
        assert engine.arms == (ArmSpec(0, 2),)
