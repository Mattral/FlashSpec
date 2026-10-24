"""Batch-1 fast path: one host read per step, EOS stopping, seeded generator.

The output-distribution checks for the fast path are in
``tests/integration/test_output_distribution_fast_path.py``. Token
parity with the previous engine is ``tests/unit/test_cache_parity.py``
plus the recorded cases in the PR. This file covers the bookkeeping.
"""

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
from flashspec.metrics.instrument import count_host_syncs
from flashspec.metrics.latency import LatencyTracker
from flashspec.sampling.greedy import greedy_step
from flashspec.sampling.rejection import rejection_step
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

_VOCAB = 512


def _llama(layers: int, seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=None,
        initializer_range=0.1,
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(scope="module")
def models() -> tuple[LlamaForCausalLM, LlamaForCausalLM]:
    """Build one (drafter, target) pair for the module."""
    return _llama(2, 1), _llama(4, 0)


def _engine(
    models: tuple[LlamaForCausalLM, LlamaForCausalLM],
    *,
    temperature: float = 1.0,
    seed: int | None = None,
    eos: tuple[int, ...] = (),
    gamma: int = 4,
    max_new: int = 40,
    profile: bool = True,
) -> SpeculativeEngine:
    drafter_model, target_model = models
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=max_new,
        eos_token_ids=eos,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=temperature, seed=seed),
        metrics=MetricsConfig(profile_steps=profile),
    )
    return SpeculativeEngine(
        config,
        [HuggingFaceDrafter("d", device="cpu", model=drafter_model)],  # type: ignore[list-item]
        TargetModel("t", torch.device("cpu"), torch.float32, model=target_model),
        UCB1Selector(n_arms=1, window_size=0),
    )


def _prompt(seed: int = 5, length: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (1, length), generator=generator)


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


class TestHostSyncs:
    """At most two device-to-host reads per step on the CPU proxy counter."""

    @pytest.mark.parametrize("temperature", [1.0, 0.7, 0.0])
    def test_reads_per_step(
        self, models: tuple[LlamaForCausalLM, LlamaForCausalLM], temperature: float
    ) -> None:
        """Sampling and greedy both read once per step: the [n, token, drafts] tolist."""
        engine = _engine(models, temperature=temperature)
        with count_host_syncs("cpu") as counter:
            result = engine.generate(_prompt())
        assert result.step_profile is not None
        steps = len(result.step_profile)
        assert counter.count <= 2 * steps, counter.by_kind
        assert counter.by_kind.get("item", 0) == 0
        assert counter.by_kind.get("__bool__", 0) == 0


class TestEos:
    """A row stops right after its first stop token; the stop token is kept."""

    @pytest.mark.parametrize(("temperature", "seed"), [(0.0, None), (1.0, 3)])
    def test_stops_after_first_eos(
        self,
        models: tuple[LlamaForCausalLM, LlamaForCausalLM],
        temperature: float,
        seed: int | None,
    ) -> None:
        """Pick a token from a full run as EOS; the stopped run is that run's prefix."""
        full = _engine(models, temperature=temperature, seed=seed).generate(_prompt())
        tokens = full.output_ids[0].tolist()
        stop_at = 17
        eos = tokens[stop_at]
        first = tokens.index(eos)
        stopped = _engine(models, temperature=temperature, seed=seed, eos=(eos,)).generate(
            _prompt()
        )
        row = stopped.output_ids[0].tolist()
        assert stopped.output_ids.shape == (1, 40)
        assert stopped.n_tokens_generated == first + 1
        assert row[: first + 1] == tokens[: first + 1]
        assert row[first + 1 :] == [-1] * (40 - first - 1)

    def test_per_call_override_and_no_eos(
        self, models: tuple[LlamaForCausalLM, LlamaForCausalLM]
    ) -> None:
        """``generate(eos_token_ids=...)`` overrides the config; ``()`` disables it."""
        engine = _engine(models, temperature=0.0)
        tokens = engine.generate(_prompt()).output_ids[0].tolist()
        eos = tokens[5]
        cut = engine.generate(_prompt(), eos_token_ids=[eos])
        assert cut.n_tokens_generated == tokens.index(eos) + 1
        configured = _engine(models, temperature=0.0, eos=(eos,))
        assert configured.generate(_prompt(), eos_token_ids=()).n_tokens_generated == 40

    def test_batch_rows_stop_independently(
        self, models: tuple[LlamaForCausalLM, LlamaForCausalLM]
    ) -> None:
        """The context-based loop stops each row at its own EOS."""
        engine = _engine(models, temperature=0.0)
        a, b = _prompt(5), _prompt(6)
        tokens_a = engine.generate(a).output_ids[0].tolist()
        tokens_b = engine.generate(b).output_ids[0].tolist()
        eos = tokens_a[9]
        result = engine.generate(torch.cat([a, b]), eos_token_ids=[eos])
        rows = result.output_ids.tolist()
        cut_a = tokens_a.index(eos) + 1
        assert rows[0][:cut_a] == tokens_a[:cut_a]
        assert set(rows[0][cut_a:]) <= {-1}
        cut_b = tokens_b.index(eos) + 1 if eos in tokens_b else 40
        assert rows[1][:cut_b] == tokens_b[:cut_b]


class TestSeed:
    """``SamplingConfig.seed`` drives a private generator, per call."""

    def test_same_seed_same_output_and_global_rng_untouched(
        self, models: tuple[LlamaForCausalLM, LlamaForCausalLM]
    ) -> None:
        """Two calls with seed 7 agree, and the global RNG state does not move."""
        engine = _engine(models, seed=7)
        torch.manual_seed(123)
        before = torch.get_rng_state()
        first = engine.generate(_prompt()).output_ids
        second = engine.generate(_prompt()).output_ids
        assert torch.equal(torch.get_rng_state(), before)
        assert torch.equal(first, second)
        other = _engine(models, seed=8).generate(_prompt()).output_ids
        assert not torch.equal(first, other)

    def test_no_seed_uses_global_rng(
        self, models: tuple[LlamaForCausalLM, LlamaForCausalLM]
    ) -> None:
        """``seed=None`` reproduces under ``torch.manual_seed``, as before."""
        engine = _engine(models)
        torch.manual_seed(4)
        first = engine.generate(_prompt()).output_ids
        torch.manual_seed(4)
        assert torch.equal(engine.generate(_prompt()).output_ids, first)

    def test_seed_in_context_based_loop(
        self, models: tuple[LlamaForCausalLM, LlamaForCausalLM]
    ) -> None:
        """Batch 2 with a seed is reproducible and leaves the global RNG alone."""
        engine = _engine(models, seed=11, max_new=12)
        batch = torch.cat([_prompt(5), _prompt(6)])
        before = torch.get_rng_state()
        first = engine.generate(batch).output_ids
        assert torch.equal(engine.generate(batch).output_ids, first)
        assert torch.equal(torch.get_rng_state(), before)


class TestStepFunctions:
    """``rejection_step`` and ``greedy_step`` edge cases."""

    def test_rejection_step_without_bonus_row(self) -> None:
        """Only gamma target rows: a full accept returns -1 as the next token."""
        lp = torch.log_softmax(torch.zeros(1, 2, 4), dim=-1)
        step = rejection_step(torch.tensor([[1, 2]]), lp, lp)
        assert step.tolist() == [[2, -1]]

    def test_rejection_step_shape_errors(self) -> None:
        """Mismatched shapes raise."""
        lp = torch.zeros(1, 2, 4)
        with pytest.raises(ValueError, match="draft_logprobs"):
            rejection_step(torch.tensor([[1, 2, 3]]), lp, lp)
        with pytest.raises(ValueError, match="target_logprobs"):
            rejection_step(torch.tensor([[1, 2]]), lp, torch.zeros(1, 4, 4))

    def test_greedy_step(self) -> None:
        """Mismatch, full accept with bonus, and full accept without a bonus row."""
        draft = torch.tensor([[1, 2], [1, 2]])
        target = torch.tensor([[1, 7, 9], [1, 2, 9]])
        assert greedy_step(draft, target).tolist() == [[1, 7], [2, 9]]
        assert greedy_step(draft, target[:, :2]).tolist() == [[1, 7], [2, -1]]


def test_latency_record() -> None:
    """Caller-measured latencies feed the percentiles; negatives raise."""
    tracker = LatencyTracker(window=10)
    for value in (1.0, 2.0, 3.0):
        tracker.record(value)
    assert tracker.p50_ms in (1.0, 2.0)
    with pytest.raises(ValueError, match="latency_ms"):
        tracker.record(-1.0)
