"""Padded verify widths cannot change a real row (CPU, fp32/fp64, tiny Llama).

Bundle ``3e38251`` had every greedy mismatch against compiled HuggingFace in
its verify-width-7 group. These tests pin down the padding side of that
question; ``tests/integration/test_width_gpu.py`` runs the same diagnostic
on the T4 target (``benchmarks/width_diagnostic.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, StaticCache

from benchmarks.width_diagnostic import _flashspec_path, _hf_paths, _path_target, pad_independence
from flashspec.engine.static_cache import StaticKvSession

_VOCAB = 256


def _llama(seed: int, *, double: bool = False) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        initializer_range=0.1,
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.pad_token_id = 0
    return model.double() if double else model


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    out = []
    for length in (5, 11, 17):
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


def test_padded_widths_are_token_exact_in_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Width 7 (full and 4-of-7 padded), width 5 and unpadded all equal HF greedy."""
    monkeypatch.setattr("benchmarks.width_diagnostic._NEW_TOKENS", 40)
    monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
    model = _llama(1)
    cache_len = 160
    paths = {
        "g6_w7": (6, _path_target(model, 7, cache_len)),
        "g2_w7": (2, _path_target(model, 7, cache_len)),
        "g4_w5": (4, _path_target(model, 5, cache_len)),
        "g6_unpadded": (6, _path_target(model, None, cache_len)),
        "g2_unpadded": (2, _path_target(model, None, cache_len)),
    }
    hf_cache = StaticCache(config=model.config, max_cache_len=cache_len)
    for prompt in _prompts():
        hf = _hf_paths(model, prompt, hf_cache)
        assert hf["hf_eager"][0] == hf["hf_static"][0]
        for name, (gamma, target) in paths.items():
            tokens, rows, widths = _flashspec_path(target, prompt, gamma)
            assert tokens == hf["hf_eager"][0], (name, prompt.shape)
            if name.endswith("_w7"):
                assert widths == [7], (name, widths)
            if name == "g2_unpadded":
                assert widths == [2, 3]
            # The row behind every emitted token predicts that token.
            for position, token in enumerate(tokens):
                assert int(rows[position].argmax()) == token, (name, position)


@pytest.mark.parametrize("width", [5, 7])
def test_other_slots_never_reach_row_zero(width: int) -> None:
    """Row 0 of a width-w call is bit-identical whatever the other w - 1 tokens are."""
    model = _llama(2)
    prefix = torch.randint(0, _VOCAB, (1, 23), generator=torch.Generator().manual_seed(4))
    result = pad_independence(model, prefix, width=width, trials=6)
    assert result["row0_bit_identical"] is True
    assert result["max_abs_difference"] == 0.0


def test_kept_cache_entries_match_the_unpadded_session() -> None:
    """After padded calls and crops, the kept keys/values equal an unpadded session's (fp64)."""
    model = _llama(5, double=True)
    plain = StaticKvSession(model, 96)
    padded = StaticKvSession(model, 96, pad_widths=(7,))
    prompt = torch.randint(0, _VOCAB, (1, 9), generator=torch.Generator().manual_seed(6))
    for session in (plain, padded):
        session.start(prompt, hold_last=True)
    generator = torch.Generator().manual_seed(7)
    for n_new, n_crop in ((2, 1), (6, 0), (3, 2), (1, 0), (4, 4)):
        ids = torch.randint(0, _VOCAB, (1, n_new), generator=generator)
        want = plain.advance(ids, keep=n_new + (1 if plain.cached_length else 0))
        got = padded.advance(ids, keep=int(want.shape[1]))
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
        for session in (plain, padded):
            session.crop(n_crop)
            session.push_pending(ids[:, -1:])
        length = plain.cached_length
        assert padded.cached_length == length
        assert int(padded._write_position()) == length  # noqa: SLF001
        for plain_layer, padded_layer in zip(
            plain._cache.layers,
            padded._cache.layers,
            strict=True,  # noqa: SLF001
        ):
            torch.testing.assert_close(
                padded_layer.keys[:, :, :length],
                plain_layer.keys[:, :, :length],
                rtol=0,
                atol=1e-12,
            )
            torch.testing.assert_close(
                padded_layer.values[:, :, :length],
                plain_layer.values[:, :, :length],
                rtol=0,
                atol=1e-12,
            )
