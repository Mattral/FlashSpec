"""``FlashSpecConfig.compile="auto"``: the decision, its log line, and both engine branches.

There is no CUDA device in CI, so the "on" branch runs the real decision
for a CUDA device with Triton assumed present, and the engine wiring is
checked on CPU with the static-cache sessions kept eager.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import flashspec.engine.speculative as speculative
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.kv_cache import KvSession
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.static_cache import StaticKvSession
from flashspec.engine.verifier import TargetModel
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig
from flashspec.utils.device import CompileDecision, resolve_compile, triton_version

_VOCAB = 128
_CUDA = torch.device("cuda", 0)  # a device object only; nothing is allocated on it
_CPU = torch.device("cpu")
_LOGGER = "flashspec.engine.speculative"


def _llama(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
    )
    return LlamaForCausalLM(config).eval()


class TestDecision:
    """resolve_compile: on with CUDA + Triton, off otherwise, always with a reason."""

    def test_default_is_auto(self) -> None:
        """FlashSpecConfig(drafter_strategy="bandit") defaults to compile='auto'."""
        assert FlashSpecConfig(drafter_strategy="bandit").compile == "auto"

    @pytest.mark.parametrize("value", [True, False, "auto"])
    def test_config_accepts(self, value: bool | str) -> None:
        """True, False and 'auto' are valid."""
        assert FlashSpecConfig(drafter_strategy="bandit", compile=value).compile == value  # type: ignore[arg-type]

    def test_config_rejects_other_strings(self) -> None:
        """Any other string is a validation error."""
        with pytest.raises(ValueError):
            FlashSpecConfig(drafter_strategy="bandit", compile="sometimes")  # type: ignore[arg-type]

    def test_auto_on_with_cuda_and_triton(self) -> None:
        """CUDA device and Triton present: on, with the Triton version in the reason."""
        decision = resolve_compile("auto", _CUDA, triton="3.1.0", check_triton=False)
        assert decision == CompileDecision(
            "auto", True, "compile=auto: CUDA device cuda:0 and Triton 3.1.0 are available"
        )

    def test_auto_off_on_cpu(self) -> None:
        """Not a CUDA device: off, and the reason names the device."""
        decision = resolve_compile("auto", _CPU, triton="3.1.0", check_triton=False)
        assert not decision.enabled
        assert "needs a CUDA device" in decision.reason
        assert "cpu" in decision.reason

    def test_auto_off_without_triton(self) -> None:
        """CUDA without Triton: auto stays off."""
        decision = resolve_compile("auto", _CUDA, triton=None, check_triton=False)
        assert not decision.enabled
        assert "Triton is not installed" in decision.reason

    def test_forced_on_without_triton_warns_in_reason(self) -> None:
        """compile=True on CUDA without Triton is honoured, with a warning reason."""
        decision = resolve_compile(True, _CUDA, triton=None, check_triton=False)
        assert decision.enabled
        assert "may fail" in decision.reason

    def test_forced_off_on_cpu(self) -> None:
        """compile=True on CPU cannot be honoured."""
        assert not resolve_compile(True, _CPU).enabled

    def test_false_is_always_off(self) -> None:
        """compile=False is off even with CUDA and Triton."""
        decision = resolve_compile(False, _CUDA, triton="3.1.0", check_triton=False)
        assert decision == CompileDecision(False, False, "compile=False")

    def test_triton_lookup_does_not_raise(self) -> None:
        """The Triton lookup returns a version or None on any machine."""
        version = triton_version()
        assert version is None or isinstance(version, str)


def _engine(compile_value: bool | str) -> tuple[SpeculativeEngine, TargetModel, list[Any]]:
    target = TargetModel("t", _CPU, torch.float32, model=_llama(1))
    drafter = HuggingFaceDrafter("d", device="cpu", model=_llama(2))
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=12,
        compile=compile_value,  # type: ignore[arg-type]
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=3, temperature=0.0),
    )
    engine = SpeculativeEngine(
        config,
        [drafter],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=1, window_size=0),
    )
    return engine, target, [drafter]


def _decision_lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith("Compiled fast path")]


class TestEngine:
    """The engine applies the decision and logs exactly one line about it."""

    def test_auto_off_branch(self, caplog: pytest.LogCaptureFixture) -> None:
        """Default config on CPU: eager dynamic cache, one INFO line saying off and why."""
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            engine, target, _ = _engine("auto")
        lines = _decision_lines(caplog)
        assert len(lines) == 1
        assert lines[0].levelno == logging.INFO
        assert lines[0].getMessage().startswith("Compiled fast path off (compile=auto needs")
        assert not engine.compiled
        assert engine.compile_decision.requested == "auto"
        assert isinstance(target._inc, KvSession)
        assert engine.generate(torch.tensor([[3, 4, 5, 6]])).n_tokens_generated == 12

    def test_auto_on_branch(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CUDA + Triton (simulated): static caches on every wrapper, one INFO line saying on."""

        def cuda_with_triton(requested: Any, device: torch.device) -> CompileDecision:  # noqa: ANN401
            assert device == _CPU
            return resolve_compile(requested, _CUDA, triton="3.1.0", check_triton=False)

        monkeypatch.setattr(speculative, "resolve_compile", cuda_with_triton)
        requests: list[bool] = []
        for cls in (TargetModel, HuggingFaceDrafter):
            original = cls.use_static_cache

            def eager_static(
                self: Any,  # noqa: ANN401
                max_cache_len: int,
                *,
                compile_forward: bool,
                _original: Any = original,  # noqa: ANN401
            ) -> None:
                requests.append(compile_forward)
                # No CUDA here: run the static cache without graph capture.
                _original(self, max_cache_len, compile_forward=False)

            monkeypatch.setattr(cls, "use_static_cache", eager_static)

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            engine, target, drafters = _engine("auto")
        lines = _decision_lines(caplog)
        assert len(lines) == 1
        assert lines[0].levelno == logging.INFO
        assert lines[0].getMessage() == (
            "Compiled fast path on (compile=auto: CUDA device cuda:0 and Triton 3.1.0 "
            "are available)."
        )
        assert engine.compiled
        assert requests == [True, True]
        assert isinstance(target._inc, StaticKvSession)
        assert isinstance(drafters[0]._inc, StaticKvSession)

        prompt = torch.tensor([[3, 4, 5, 6]])
        out = engine.generate(prompt).output_ids
        reference = target._model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=12,
            min_new_tokens=12,
        )
        assert out[0].tolist() == reference[0, prompt.shape[1] :].tolist()

    def test_false_logs_off(self, caplog: pytest.LogCaptureFixture) -> None:
        """compile=False logs one INFO line."""
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            engine, _, _ = _engine(False)
        lines = _decision_lines(caplog)
        assert [r.getMessage() for r in lines] == ["Compiled fast path off (compile=False)."]
        assert not engine.compiled

    def test_forced_on_cpu_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """compile=True on CPU logs one WARNING line and stays eager."""
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            engine, _, _ = _engine(True)
        lines = _decision_lines(caplog)
        assert len(lines) == 1
        assert lines[0].levelno == logging.WARNING
        assert not engine.compiled
