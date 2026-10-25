"""Per-arm gamma, fixed-width compiled calls, and the idle drafter's catch-up.

CUDA graphs need a GPU, so the static sessions run eagerly here
(``compile_forward=False``). Padding, cropping the padded positions, and
the shape bookkeeping are the same code with or without capture.
``StaticKvSession.graph_call_shapes`` counts the (width, rows) keys that
the captured path would see; one CUDA graph is recorded per key.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.base import DraftSelector
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import ArmSpec, arm_label, default_arms, validate_arms
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import DRAFTER_PAD_WIDTHS, SpeculativeEngine
from flashspec.engine.static_cache import StaticKvSession
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.acceptance import AcceptanceTracker
from flashspec.metrics.instrument import ForwardRecorder
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

_VOCAB = 256
_NEW = 40


def _llama(layers: int, seed: int, *, double: bool = False) -> LlamaForCausalLM:
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
    model = LlamaForCausalLM(config).eval()
    return model.double() if double else model


def _perturbed(seed: int, scale: float) -> LlamaForCausalLM:
    model = _llama(3, seed)
    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)
    return model


class _Scripted(DraftSelector):
    """Plays a fixed arm sequence (cyclically) and records every update."""

    def __init__(self, n_arms: int, script: Sequence[int]) -> None:
        super().__init__(n_arms=n_arms, window_size=0)
        self._script = list(script)
        self.updates: list[tuple[int, int, float | None]] = []

    def select(self) -> int:
        return self._script[self._t % len(self._script)]

    def update(self, arm: int, accepted: int, reward: float | None = None) -> None:
        self.updates.append((arm, accepted, reward))
        self._arms[arm].record(accepted, reward=reward)
        self._t += 1

    def _state_dict(self) -> dict[str, Any]:
        return {"type": "scripted"}

    @classmethod
    def _from_state_dict(cls, state: dict[str, Any]) -> _Scripted:
        raise NotImplementedError


def _engine(
    target: TargetModel,
    drafters: list[object],
    arms: list[ArmSpec],
    *,
    temperature: float = 0.0,
    bandit: DraftSelector | None = None,
    seed: int | None = None,
    max_new: int = _NEW,
) -> SpeculativeEngine:
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=max_new,
        bandit=BanditConfig(n_arms=len(arms), strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=4, temperature=temperature, seed=seed),
        metrics=MetricsConfig(profile_steps=True),
    )
    return SpeculativeEngine(
        config,
        drafters,  # type: ignore[arg-type]
        target,
        bandit or UCB1Selector(n_arms=len(arms), window_size=0),
        arms=arms,
    )


def _static(owner: Any, widths: Sequence[int] | None) -> None:  # noqa: ANN401
    owner.use_static_cache(64, compile_forward=False, pad_widths=widths)


def _hf_greedy(model: LlamaForCausalLM, prompt: torch.Tensor, n: int = _NEW) -> torch.Tensor:
    with torch.no_grad():
        out = model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=n,
            min_new_tokens=n,
            eos_token_id=None,
            pad_token_id=0,
        )
    return out[:, prompt.shape[1] :]


def _prompt(seed: int, length: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    chunk = torch.randint(0, _VOCAB, (1, length), generator=generator)
    return torch.cat([chunk, chunk[:, : length // 2]], dim=1)


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


class TestArmSpec:
    """Arm validation and defaults."""

    def test_defaults_and_labels(self) -> None:
        """One arm per drafter at one gamma; unlabelled arms get arm<k>."""
        arms = default_arms(2, 3)
        assert arms == [ArmSpec(0, 3), ArmSpec(1, 3)]
        assert arm_label(arms, 1) == "arm1"
        assert arm_label([ArmSpec(0, 2, label="qwen-g2")], 0) == "qwen-g2"

    def test_validation(self) -> None:
        """Bad drafter index, gamma below 1, no arms, or a count mismatch raise."""
        with pytest.raises(ValueError, match="names drafter"):
            validate_arms([ArmSpec(2, 4)], 2)
        with pytest.raises(ValueError, match="gamma"):
            validate_arms([ArmSpec(0, 0)], 1)
        with pytest.raises(ValueError, match="At least one"):
            validate_arms([], 1)
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(1, 1))
        with pytest.raises(ValueError, match="len\\(arms\\)"):
            SpeculativeEngine(
                FlashSpecConfig(
                    drafter_strategy="bandit", device="cpu", bandit=BanditConfig(n_arms=2)
                ),
                [NgramDrafter()],  # type: ignore[list-item]
                target,
                UCB1Selector(n_arms=2),
                arms=[ArmSpec(0, 2)],
            )

    def test_acceptance_tracker_counts_proposed(self) -> None:
        """The rate divides by what was proposed; accepting more than that raises."""
        tracker = AcceptanceTracker(gamma=6)
        tracker.record(2, 2)
        tracker.record(1, 4)
        tracker.record(0, 0)
        assert tracker.mean_acceptance_rate == 3 / 6
        with pytest.raises(ValueError, match="n_accepted"):
            tracker.record(3, 2)
        with pytest.raises(ValueError, match="n_proposed"):
            tracker.record(0, 7)


class TestPerArmGamma:
    """Arms sharing one drafter at different gammas."""

    @pytest.mark.parametrize("static", [False, True])
    def test_greedy_exact_and_verify_lengths(self, static: bool) -> None:
        """Each step forwards pending + that arm's gamma (dynamic) or the fixed width (static)."""
        target_model = _llama(3, 10)
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
        drafter = HuggingFaceDrafter("d", device="cpu", model=_perturbed(10, 0.003))
        arms = [ArmSpec(0, 2), ArmSpec(0, 4), ArmSpec(0, 6)]
        if static:
            _static(target, (7,))
            _static(drafter, DRAFTER_PAD_WIDTHS)
        script = [0, 1, 2, 2, 0, 1]
        engine = _engine(target, [drafter], arms, bandit=_Scripted(3, script))
        prompt = _prompt(3, 8)
        with ForwardRecorder(target_model) as recorder:
            result = engine.generate(prompt)
        assert torch.equal(result.output_ids, _hf_greedy(target_model, prompt))
        assert result.step_profile is not None
        n_steps = len(result.step_profile)
        gammas = [arms[script[i % len(script)]].gamma for i in range(n_steps)]
        lengths = recorder.lengths
        assert lengths[0] == int(prompt.shape[1])
        if static:
            assert lengths[1:] == [7] * n_steps
        else:
            assert lengths[1:] == [gammas[0]] + [g + 1 for g in gammas[1:]]
        for step, gamma in zip(result.step_profile, gammas, strict=True):
            assert step.tokens_emitted <= gamma + 1

    def test_sampled_tokens_match_unpadded(self) -> None:
        """Padding changes nothing: same seed, same tokens as unpadded static (float64)."""
        target_model = _llama(3, 20, double=True)
        drafter_model = _perturbed(20, 0.01).double()
        arms = [ArmSpec(0, 2), ArmSpec(0, 5), ArmSpec(1, 3)]
        outputs = []
        for widths in ((None, None), ((6,), DRAFTER_PAD_WIDTHS)):
            target = TargetModel("t", torch.device("cpu"), torch.float64, model=target_model)
            drafter = HuggingFaceDrafter("d", device="cpu", model=drafter_model)
            _static(target, widths[0])
            _static(drafter, widths[1])
            engine = _engine(
                target,
                [drafter, NgramDrafter()],
                arms,
                temperature=0.8,
                seed=4,
                bandit=_Scripted(3, [0, 2, 1, 2, 2, 0]),
            )
            outputs.append(engine.generate(_prompt(4, 10)).output_ids)
        assert torch.equal(outputs[0], outputs[1])


class TestPaddedSession:
    """``StaticKvSession`` with pad widths."""

    def test_padded_rows_equal_unpadded_and_are_cropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real rows match the unpadded call; the write position counts real tokens only."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        model = _llama(2, 30, double=True)
        plain = StaticKvSession(model, 64)
        padded = StaticKvSession(model, 64, pad_widths=(1, 2, 8))
        prompt = torch.randint(0, _VOCAB, (1, 9), generator=torch.Generator().manual_seed(1))
        context = prompt
        for session in (plain, padded):
            session.start(prompt, hold_last=False)
        for n, keep in ((3, 3), (1, 1), (2, 1), (5, 2), (8, 8)):
            ids = torch.randint(0, _VOCAB, (1, n), generator=torch.Generator().manual_seed(n))
            want = plain.advance(ids, keep=keep)
            got = padded.advance(ids, keep=keep)
            assert got.shape == (1, keep, _VOCAB)
            torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)
            context = torch.cat([context, ids], dim=1)
            padded.check(context)
            assert padded.cached_length == int(context.shape[1])
        # Every graph-path call used one of the listed widths, all rows kept.
        assert set(padded.graph_call_shapes) == {(8, 8), (1, 1), (2, 2)}

    def test_long_call_runs_unpadded_and_reserve_leaves_room(self) -> None:
        """A call wider than every width is not padded; reserve adds the widest pad."""
        model = _llama(2, 31)
        session = StaticKvSession(model, 16, pad_widths=(4,))
        session.reserve(20)
        assert session._max_len == 24  # noqa: SLF001
        session.start(torch.zeros((1, 3), dtype=torch.long), hold_last=False)
        session.advance(torch.ones((1, 6), dtype=torch.long), keep=1)
        assert session.cached_length == 9
        assert session.graph_call_shapes == {}
        with pytest.raises(ValueError, match="positive"):
            session.set_pad_widths([0])
        session.set_pad_widths(None)
        assert session.pad_widths == ()

    def test_use_static_cache_is_idempotent(self) -> None:
        """A second call keeps the session (and its graphs) and merges widths."""
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(1, 32))
        _static(target, (5,))
        first = target._inc  # noqa: SLF001
        _static(target, (7,))
        assert target._inc is first  # noqa: SLF001
        assert first.pad_widths == (5, 7)


class TestPaddingNeverAccepted:
    """Filler and padded positions are never accepted, even where the target agrees."""

    @staticmethod
    def _zero_head(model: LlamaForCausalLM) -> LlamaForCausalLM:
        # All logits 0: the target's argmax is token 0 everywhere, so a
        # filler of 0 would match it if the length mask did not stop it.
        with torch.no_grad():
            model.lm_head.weight.zero_()
        return model

    def test_lookup_filler_and_short_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Lookup proposes 1 real token on an all-zero context; gamma-2 arm emits at most 3."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        target_model = self._zero_head(_llama(2, 40))
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
        drafter = HuggingFaceDrafter("d", device="cpu", model=self._zero_head(_llama(1, 41)))
        _static(target, (7,))
        _static(drafter, DRAFTER_PAD_WIDTHS)
        arms = [ArmSpec(1, 6), ArmSpec(0, 2)]
        bandit = _Scripted(2, [0, 0, 1, 0, 1])
        engine = _engine(target, [drafter, NgramDrafter()], arms, bandit=bandit, max_new=30)
        prompt = torch.zeros((1, 5), dtype=torch.long)
        result = engine.generate(prompt)
        assert torch.equal(result.output_ids, torch.zeros((1, 30), dtype=torch.long))
        assert result.step_profile is not None
        for index, step in enumerate(result.step_profile[:-1]):
            arm = bandit.updates[index][0]
            # Lookup on all zeros: the most recent earlier 3-gram ends one
            # token before the end, so exactly one real proposal (+ 1 target
            # token). The drafter arm: its 2 drafts + bonus, never the width 7.
            assert step.tokens_emitted == (2 if arm == 0 else 3), (index, arm)


class TestCatchUp:
    """An idle model drafter forwards everything it missed in one call, charged to its step."""

    @pytest.mark.parametrize("static", [False, True])
    def test_one_call_of_all_missing_tokens(
        self, static: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After k lookup steps the drafter's next forward carries all of them at once."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        target_model = _llama(3, 50)
        drafter_model = _perturbed(50, 0.003)
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
        drafter = HuggingFaceDrafter("d", device="cpu", model=drafter_model)
        if static:
            _static(target, (5,))
            _static(drafter, DRAFTER_PAD_WIDTHS)
        arms = [ArmSpec(0, 4), ArmSpec(1, 4)]
        script = [0, 1, 1, 1, 0, 1, 0]
        engine = _engine(target, [drafter, NgramDrafter()], arms, bandit=_Scripted(2, script))
        prompt = _prompt(5, 12)
        with ForwardRecorder(drafter_model) as recorder:
            result = engine.generate(prompt)
        assert torch.equal(result.output_ids, _hf_greedy(target_model, prompt))
        assert result.step_profile is not None
        emitted = [step.tokens_emitted for step in result.step_profile]
        arm_of = [script[i % len(script)] for i in range(len(emitted))]
        # Rebuild the expected drafter forwards: prefill, then per model-arm
        # step one call carrying its pending tokens, then gamma - 1 one-token
        # calls. Pending = the drafter's own leftover (1 after a rejection,
        # 2 after a full accept) plus every token lookup steps emitted since.
        expected: list[int] = [int(prompt.shape[1]) - 1]
        pending = 1
        for count, arm in zip(emitted, arm_of, strict=True):
            if arm == 1:
                pending += count
                continue
            width = pending
            if static and pending <= DRAFTER_PAD_WIDTHS[-1]:
                width = next(w for w in DRAFTER_PAD_WIDTHS if w >= pending)
            expected += [width, 1, 1, 1]
            pending = 2 if count == 5 else 1
        assert recorder.lengths == expected
        assert any(width > 2 for width in expected[1:]), expected

    def test_catch_up_cost_is_charged_to_the_model_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fake clock: the catch-up forward's time is in the model arm's step reward."""
        target_model = _llama(2, 60)
        drafter_model = _perturbed(60, 0.003)
        clock = {"now": 0.0}
        drafter_calls: list[int] = []

        def _tick_drafter(_module: object, args: tuple[Any, ...], _out: object) -> None:
            n = int(args[0].shape[1]) if args else 0
            drafter_calls.append(n)
            clock["now"] += 1.0 + 0.5 * n

        def _tick_target(_module: object, _args: object, _out: object) -> None:
            clock["now"] += 3.0

        handles = [
            drafter_model.register_forward_hook(_tick_drafter),
            target_model.register_forward_hook(_tick_target),
        ]
        monkeypatch.setattr(time, "perf_counter", lambda: clock["now"])
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
        drafter = HuggingFaceDrafter("d", device="cpu", model=drafter_model)
        # Model arm, three lookup steps, then the model arm again (the catch-up step).
        bandit = _Scripted(2, [0, 1, 1, 1, 0])
        engine = _engine(
            target, [drafter, NgramDrafter()], [ArmSpec(0, 3), ArmSpec(1, 3)], bandit=bandit
        )
        try:
            result = engine.generate(_prompt(6, 10))
        finally:
            for handle in handles:
                handle.remove()
        assert result.step_profile is not None
        emitted = [step.tokens_emitted for step in result.step_profile]
        # Step 0: drafter prefill + 3 draft forwards. Step 4: one catch-up
        # forward (its leftover pending plus all tokens of steps 1-3), then
        # 2 one-token forwards. Its clock: those 3 calls + the target's 3.0.
        catch_up = drafter_calls[4:7]
        leftover = 2 if emitted[0] == 4 else 1
        assert catch_up[0] == leftover + sum(emitted[1:4])
        assert catch_up[1:] == [1, 1]
        cost = 3.0 + sum(1.0 + 0.5 * n for n in catch_up)
        _arm, _accepted, reward = bandit.updates[4]
        assert reward == pytest.approx(emitted[4] / cost)
        # Lookup steps cost only the target forward on this clock.
        for index in (1, 2, 3):
            assert bandit.updates[index][2] == pytest.approx(emitted[index] / 3.0)
