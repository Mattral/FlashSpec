"""Step profiler, forward recorder, and host-sync counter."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import ForwardRecorder, count_host_syncs
from flashspec.metrics.step_profile import PHASES, StepProfiler, StepRecord
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig


def _llama(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=None,
    )
    return LlamaForCausalLM(config).eval()


def _engine(profile: bool) -> SpeculativeEngine:
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=12,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=3),
        metrics=MetricsConfig(profile_steps=profile),
    )
    return SpeculativeEngine(
        config,
        [HuggingFaceDrafter("draft", device="cpu", model=_llama(1))],  # type: ignore[list-item]
        TargetModel("target", torch.device("cpu"), torch.float32, model=_llama(2)),
        UCB1Selector(n_arms=1, window_size=0),
    )


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


class TestStepProfiler:
    """Phase marks become per-step records and a summary."""

    def test_records_and_summary(self) -> None:
        """Two steps give two records and fractions that sum to one."""
        profiler = StepProfiler("cpu")
        for tokens in (2, 4):
            profiler.start_step()
            for phase in PHASES:
                profiler.mark(phase)
            profiler.end_step(tokens_emitted=tokens)
        records = profiler.records()
        assert len(records) == 2
        assert [record.tokens_emitted for record in records] == [2, 4]
        assert all(record.total_ms >= 0.0 for record in records)
        summary = profiler.summary()
        assert summary["n_steps"] == 2.0
        assert summary["tokens_per_step_mean"] == 3.0
        fractions = sum(summary[f"{phase}_fraction"] for phase in PHASES)
        assert fractions == pytest.approx(1.0) or summary["step_ms_mean"] == 0.0

    def test_unmarked_phase_is_zero_and_reset_clears(self) -> None:
        """A phase that was never marked is zero; reset drops everything."""
        profiler = StepProfiler("cpu")
        profiler.start_step()
        profiler.mark("verify")
        profiler.end_step(tokens_emitted=1)
        (record,) = profiler.records()
        assert record.draft_ms == 0.0
        assert record.sample_ms == 0.0
        profiler.reset()
        assert profiler.records() == []
        assert profiler.summary() == {}

    def test_misuse_raises(self) -> None:
        """Marks outside a step and unknown phase names raise."""
        profiler = StepProfiler("cpu")
        with pytest.raises(RuntimeError):
            profiler.mark("draft")
        with pytest.raises(RuntimeError):
            profiler.end_step(tokens_emitted=0)
        profiler.start_step()
        with pytest.raises(ValueError, match="Unknown phase"):
            profiler.mark("prefill")

    def test_engine_profile_is_off_by_default(self) -> None:
        """The default config does not profile."""
        result = _engine(profile=False).generate(torch.randint(0, 64, (1, 5)))
        assert result.step_profile is None

    def test_engine_profile_counts_every_emitted_token(self) -> None:
        """Per-step token counts add up to the generated length."""
        result = _engine(profile=True).generate(torch.randint(0, 64, (1, 5)))
        assert result.step_profile is not None
        assert all(isinstance(step, StepRecord) for step in result.step_profile)
        assert sum(step.tokens_emitted for step in result.step_profile) == 12
        assert all(step.verify_ms > 0.0 for step in result.step_profile)


class TestForwardRecorder:
    """Positional and keyword ``input_ids`` are both recorded."""

    def test_lengths_and_detach(self) -> None:
        """Lengths are recorded while attached and kept after detach."""
        model = _llama(0)
        recorder = ForwardRecorder(model).attach()
        recorder.attach()  # A second attach keeps one hook.
        with torch.no_grad():
            model(torch.zeros(1, 3, dtype=torch.long))
            model(input_ids=torch.zeros(1, 2, dtype=torch.long))
        recorder.detach()
        with torch.no_grad():
            model(torch.zeros(1, 4, dtype=torch.long))
        assert recorder.lengths == [3, 2]
        assert recorder.calls == 2
        recorder.clear()
        assert recorder.lengths == []


class TestHostSyncCounter:
    """The CPU counter sees each read and restores the tensor class."""

    def test_counts_reads_and_restores(self) -> None:
        """Five reads are counted; the class dict is restored afterwards."""
        before = {name: name in vars(torch.Tensor) for name in ("item", "__bool__")}
        values = torch.tensor([1.0, 2.0])
        with count_host_syncs("cpu") as counter:
            values.sum().item()
            bool(values.any())
            values.tolist()
            int(values[0])
            float(values[1])
            values.add(1.0)  # Not a host read.
        assert counter.count == 5
        assert counter.by_kind == {
            "item": 1,
            "__bool__": 1,
            "tolist": 1,
            "__int__": 1,
            "__float__": 1,
        }
        after = {name: name in vars(torch.Tensor) for name in ("item", "__bool__")}
        assert after == before
        assert values.sum().item() == 3.0

    def test_engine_step_sync_count_is_bounded(self) -> None:
        """The batch-1 loop reads the device at most twice per step (once, in fact)."""
        engine = _engine(profile=True)
        prompt = torch.randint(0, 64, (1, 5))
        with count_host_syncs("cpu") as counter:
            result = engine.generate(prompt)
        assert result.step_profile is not None
        assert counter.count > 0
        assert counter.count / len(result.step_profile) <= 2.0
