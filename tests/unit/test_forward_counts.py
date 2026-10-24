"""Forward calls per speculative step, counted on a tiny random Llama.

Speed at batch 1 is set by how many forwards each model runs per step,
not by how many tokens each forward carries. These tests record the
``input_ids`` length of every forward and pin the per-step pattern.

Pattern pinned here:

- Both models prefill ``prompt[:-1]``; the last prompt token is pending.
- Target: exactly one forward per step, of the pending token plus the
  ``gamma`` draft tokens (``1 + gamma``). Committing the residual or
  bonus token queues it as the next pending token and never forwards.
- Drafter: exactly ``gamma`` forwards per step. The first carries the
  pending tokens: one after a rejection, two after a full accept (the
  last draft token, never forwarded, then the bonus token). The others
  carry one draft token each.

Before this pattern the target re-forwarded the residual or bonus token
alone after every verify (two target forwards per step), and the
drafter ran ``gamma + 1`` forwards per step.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.speculative import GenerationResult, SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import ForwardRecorder
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

_VOCAB = 512
_PROMPT_LEN = 8
_MAX_NEW = 40


def _llama(layers: int, seed: int) -> LlamaForCausalLM:
    """Random tiny Llama. ``initializer_range`` 0.1 mixes rejections and full accepts."""
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


def _run(gamma: int, seed: int) -> tuple[GenerationResult, list[int], list[int]]:
    """Generate once and return the result plus target and drafter forward lengths."""
    target_model = _llama(4, seed)
    drafter_model = _llama(2, seed + 1)
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=_MAX_NEW,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=1.0),
        metrics=MetricsConfig(profile_steps=True),
    )
    engine = SpeculativeEngine(
        config,
        [HuggingFaceDrafter("draft", device="cpu", model=drafter_model)],  # type: ignore[list-item]
        TargetModel("target", torch.device("cpu"), torch.float32, model=target_model),
        UCB1Selector(n_arms=1, window_size=0),
    )
    torch.manual_seed(seed + 100)
    prompt = torch.randint(0, _VOCAB, (1, _PROMPT_LEN))
    with ForwardRecorder(target_model) as target, ForwardRecorder(drafter_model) as drafter:
        result = engine.generate(prompt, max_new_tokens=_MAX_NEW)
    return result, target.lengths, drafter.lengths


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


_CASES = [(4, 0), (4, 7), (2, 3), (1, 5), (6, 9)]


@pytest.mark.parametrize(("gamma", "seed"), _CASES)
def test_target_forwards_per_step(gamma: int, seed: int) -> None:
    """One prefill, then exactly one ``1 + gamma`` forward per step."""
    result, target_lengths, _ = _run(gamma, seed)
    assert result.step_profile is not None
    n_steps = len(result.step_profile)
    assert result.n_tokens_generated == _MAX_NEW
    assert target_lengths[0] == _PROMPT_LEN - 1
    assert target_lengths[1:] == [1 + gamma] * n_steps, target_lengths
    assert len(target_lengths) == 1 + n_steps


@pytest.mark.parametrize(("gamma", "seed"), _CASES)
def test_drafter_forwards_per_step(gamma: int, seed: int) -> None:
    """One prefill, then exactly ``gamma`` forwards per step; the first carries the pending."""
    result, _, drafter_lengths = _run(gamma, seed)
    assert result.step_profile is not None
    emitted = [step.tokens_emitted for step in result.step_profile]
    expected = [_PROMPT_LEN - 1]
    for index in range(len(emitted)):
        full_accept_before = index > 0 and emitted[index - 1] == gamma + 1
        expected += [2 if full_accept_before else 1] + [1] * (gamma - 1)
    assert drafter_lengths == expected, (drafter_lengths, emitted)
    assert len(drafter_lengths) == 1 + gamma * len(emitted)


def test_run_covers_rejections_and_full_accepts() -> None:
    """The toy pair exercises both cache paths: a rejection and a full accept plus bonus."""
    gamma = 4
    result, _, _ = _run(gamma, 0)
    assert result.step_profile is not None
    emitted = [step.tokens_emitted for step in result.step_profile]
    assert gamma + 1 in emitted, emitted
    assert any(count < gamma + 1 for count in emitted[:-1]), emitted
