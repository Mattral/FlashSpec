"""The KV-cached engine emits the same tokens as the cache-free engine.

Both paths consume the global RNG in the same order (``gamma`` draft
draws, the acceptance uniforms, then a residual or bonus draw), so with
a fixed seed they must produce identical tokens. The models run in
float64 so that the cached and cache-free forwards agree far below the
float32 resolution the engine samples at.

This is the determinism check for any change to the cache bookkeeping:
the cache-free path never crops or replays, so it is the reference.
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
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig

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
    return LlamaForCausalLM(config).double().eval()


def _generate(gamma: int, temperature: float, seed: int, cached: bool) -> torch.Tensor:
    drafter = HuggingFaceDrafter(
        "draft", device="cpu", temperature=temperature, model=_llama(2, seed + 1)
    )
    target = TargetModel("target", torch.device("cpu"), torch.float64, model=_llama(4, seed))
    if not cached:
        drafter._kv.disable()
        target._kv.disable()
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=48,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=temperature),
    )
    engine = SpeculativeEngine(
        config,
        [drafter],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=1, window_size=0),
    )
    torch.manual_seed(seed + 100)
    prompt = torch.randint(0, _VOCAB, (1, 8))
    torch.manual_seed(seed + 200)
    return engine.generate(prompt).output_ids


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


@pytest.mark.parametrize(
    ("gamma", "temperature", "seed"),
    [(4, 1.0, 0), (4, 1.0, 1), (4, 0.7, 2), (2, 1.0, 3), (1, 1.0, 4), (6, 1.3, 5)],
)
def test_cached_engine_matches_cache_free_engine(gamma: int, temperature: float, seed: int) -> None:
    """Same seed, same tokens, with and without the KV cache."""
    cached = _generate(gamma, temperature, seed, cached=True)
    reference = _generate(gamma, temperature, seed, cached=False)
    assert cached.shape == (1, 48)
    assert torch.equal(cached, reference), (cached.tolist(), reference.tolist())
