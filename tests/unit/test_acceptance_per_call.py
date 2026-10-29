"""GenerationResult.acceptance_rate covers one generate() call only.

Up to 0.3.1 the engine's acceptance tracker was never reset, so each call
reported a running mean over every earlier call on the same engine (warm-ups
included). Each call's rate must equal its own accepted / proposed counts.
Toy models on CPU: correctness only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import torch

from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.speculative import GenerationResult, SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.utils.config import FlashSpecConfig, SamplingConfig
from tests.unit.test_composite_default import _NEW, _llama, _perturbed, _prompts


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _engine(temperature: float = 0.0) -> SpeculativeEngine:
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=_NEW,
        sampling=SamplingConfig(gamma=3, temperature=temperature),
    )
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(3, 10))
    drafter = HuggingFaceDrafter("d", device="cpu", model=_perturbed(10, 0.05))
    return SpeculativeEngine(config, [drafter], target)


def _own_rate(result: GenerationResult) -> float:
    """Return accepted over proposed draft tokens, from this call's step records."""
    accepted = sum(step.accepted for step in result.arm_steps)
    proposed = sum(step.proposed for step in result.arm_steps)
    return accepted / proposed if proposed else 0.0


def test_batch_one_calls_report_their_own_rate() -> None:
    """Batch 1 (incremental path): every call's rate is its own, and a repeat call matches."""
    engine = _engine()
    prompts = _prompts()
    results = [engine.generate(prompt) for prompt in [*prompts, prompts[0]]]
    own = [_own_rate(result) for result in results]
    # The prompts must differ in acceptance, or a running mean would pass too.
    assert len(set(own)) > 1, own
    assert [result.acceptance_rate for result in results] == own
    # Greedy: the same prompt again gives the same steps and the same rate.
    assert results[-1].acceptance_rate == results[0].acceptance_rate


def test_batched_calls_report_their_own_rate() -> None:
    """Batch 2 (the general step loop): same rule after batch-1 calls on the same engine."""
    engine = _engine()
    prompts = _prompts()
    first = engine.generate(prompts[2])
    batch = engine.generate(torch.cat([prompts[0], prompts[0]], dim=0))
    assert batch.n_tokens_generated == 2 * _NEW
    assert first.acceptance_rate == _own_rate(first)
    assert batch.acceptance_rate == _own_rate(batch)
    assert batch.acceptance_rate != first.acceptance_rate


def test_sampled_calls_report_their_own_rate() -> None:
    """Temperature 1.0: the rule holds when sampling too."""
    engine = _engine(temperature=1.0)
    for prompt in _prompts():
        result = engine.generate(prompt)
        assert result.acceptance_rate == _own_rate(result)
