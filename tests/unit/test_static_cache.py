"""Static-cache incremental path (``FlashSpecConfig.compile``), checked eagerly on CPU.

CUDA-graph capture needs a GPU. Everything around it, the static cache,
moving the write position instead of cropping, device-side
``cache_position``, reuse across calls, is the same code with or
without capture, so these tests run it eagerly (``compile_forward=False``)
and compare with the dynamic cache, HuggingFace ``generate`` and the
cache-free engine.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.kv_cache import CacheUnavailableError, KvSession
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.static_cache import StaticKvSession
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import ForwardRecorder
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

_VOCAB = 256


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


def _perturbed(seed: int, scale: float, *, double: bool = False) -> LlamaForCausalLM:
    model = _llama(3, seed, double=double)
    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator) * scale
            parameter.add_(noise.to(parameter.dtype))
    return model


def _engine(
    target_model: LlamaForCausalLM,
    drafter_models: list[LlamaForCausalLM],
    *,
    temperature: float,
    gamma: int = 4,
    max_new: int = 40,
    static: bool = True,
    cache_len: int = 64,
    seed: int | None = None,
) -> SpeculativeEngine:
    dtype = next(target_model.parameters()).dtype
    target = TargetModel("t", torch.device("cpu"), dtype, model=target_model)
    drafters = [
        HuggingFaceDrafter(f"d{i}", device="cpu", model=model)
        for i, model in enumerate(drafter_models)
    ]
    if static:
        for owner in (target, *drafters):
            owner.use_static_cache(cache_len, compile_forward=False)
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=max_new,
        bandit=BanditConfig(n_arms=len(drafters), strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=temperature, seed=seed),
        metrics=MetricsConfig(profile_steps=True),
    )
    return SpeculativeEngine(
        config,
        drafters,  # type: ignore[arg-type]
        target,
        UCB1Selector(n_arms=len(drafters), window_size=0),
    )


def _prompt(seed: int, length: int) -> torch.Tensor:
    return torch.randint(0, _VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


class TestSession:
    """StaticKvSession gives the dynamic session's logits for the same operations."""

    def test_matches_dynamic_session(self) -> None:
        """Start / advance / crop / push_pending / advance, both sessions, same rows."""
        model = _llama(2, 0)
        prompt = _prompt(1, 9)
        dynamic, static = KvSession(model), StaticKvSession(model, 32)
        out: dict[str, list[torch.Tensor]] = {"d": [], "s": []}
        for key, session in (("d", dynamic), ("s", static)):
            first = session.start(prompt, hold_last=False)
            assert first is not None
            out[key].append(first)
            out[key].append(session.advance(torch.tensor([[5, 6, 7, 8]]), keep=4))
            session.crop(2)
            session.push_pending(torch.tensor([[9]]))
            out[key].append(session.advance(torch.tensor([[10, 11, 12, 13]]), keep=5))
            assert session.cached_length == 9 + 4 - 2 + 5
        for dyn, sta in zip(out["d"], out["s"], strict=True):
            assert torch.allclose(dyn, sta, rtol=1e-4, atol=1e-5)

    def test_hold_last_and_one_token_prompt(self) -> None:
        """The drafter mode keeps the last token pending; a 1-token prompt needs no forward."""
        model = _llama(1, 2)
        session = StaticKvSession(model, 16)
        with ForwardRecorder(model) as recorder:
            assert session.start(torch.tensor([[4]]), hold_last=True) is None
        assert recorder.lengths == []
        assert session.pending_length == 1
        assert session.advance(None, keep=1).shape == (1, 1, _VOCAB)

    def test_guards_reserve_and_reset(self) -> None:
        """Full cache raises; reserve grows it; reset rewinds without reallocation."""
        model = _llama(1, 3)
        session = StaticKvSession(model, 16)
        session.start(_prompt(2, 12), hold_last=False)
        with pytest.raises(CacheUnavailableError, match="full"):
            session.advance(torch.zeros(1, 8, dtype=torch.long), keep=1)
        session.reserve(64)
        session.start(_prompt(2, 12), hold_last=False)
        session.advance(torch.zeros(1, 8, dtype=torch.long), keep=1)
        assert session.cached_length == 20
        cache = session._cache
        session.reset()
        assert session.cached_length == 0
        assert session._cache is cache
        with pytest.raises(CacheUnavailableError, match="cannot crop"):
            session.crop(3)
        assert not session.enabled
        assert not session.compiled

    def test_debug_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The assertion path compares the write position and the ids."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        session = StaticKvSession(_llama(1, 4), 16)
        prompt = _prompt(3, 5)
        session.start(prompt, hold_last=True)
        session.check(prompt)
        with pytest.raises(AssertionError, match="do not match"):
            session.check((prompt + 1) % _VOCAB)


class TestEngineOnStaticCache:
    """The engine on static caches: exact greedy, same sampled tokens, same forward pattern."""

    @pytest.mark.parametrize("gamma", [1, 4, 6])
    def test_greedy_matches_hf_generate(self, gamma: int) -> None:
        """Token for token against generate(do_sample=False), several prompts and lengths."""
        target_model = _llama(3, 10)
        engine = _engine(target_model, [_perturbed(10, 0.003)], temperature=0.0, gamma=gamma)
        for seed, length in ((1, 1), (2, 7), (3, 15)):
            prompt = _prompt(seed, length)
            with torch.no_grad():
                expected = target_model.generate(
                    prompt,
                    attention_mask=torch.ones_like(prompt),
                    do_sample=False,
                    max_new_tokens=40,
                    min_new_tokens=40,
                    eos_token_id=None,
                    pad_token_id=0,
                )[:, length:]
            assert torch.equal(engine.generate(prompt).output_ids, expected), (gamma, seed)

    @pytest.mark.parametrize(("gamma", "temperature"), [(4, 1.0), (2, 0.7)])
    def test_sampled_tokens_match_cache_free_engine(self, gamma: int, temperature: float) -> None:
        """Same seed, same tokens as the cache-free engine (float64 models)."""
        target_model = _llama(3, 20, double=True)
        drafter_model = _perturbed(20, 0.01, double=True)
        prompt = _prompt(5, 8)
        static = _engine(
            target_model, [drafter_model], temperature=temperature, gamma=gamma, seed=11
        )
        free = _engine(
            target_model,
            [drafter_model],
            temperature=temperature,
            gamma=gamma,
            static=False,
            seed=11,
        )
        free._target._kv.disable()
        free._drafters[0]._kv.disable()  # type: ignore[attr-defined]
        assert torch.equal(static.generate(prompt).output_ids, free.generate(prompt).output_ids)

    def test_forward_pattern_and_reuse_across_calls(self) -> None:
        """Same forward pattern as the dynamic cache, and a second call reuses the cache."""
        target_model, drafter_model = _llama(3, 30), _perturbed(30, 0.003)
        engine = _engine(target_model, [drafter_model], temperature=0.0, gamma=4)
        prompt = _prompt(6, 10)
        with ForwardRecorder(target_model) as target_rec, ForwardRecorder(drafter_model) as d_rec:
            first = engine.generate(prompt)
        assert first.step_profile is not None
        n_steps = len(first.step_profile)
        assert target_rec.lengths == [10, 4] + [5] * (n_steps - 1)
        assert len(d_rec.lengths) == 1 + 4 * n_steps
        cache = engine._target._inc._cache  # type: ignore[attr-defined]
        second = engine.generate(prompt)
        assert torch.equal(first.output_ids, second.output_ids)
        assert engine._target._inc._cache is cache  # type: ignore[attr-defined]

    def test_long_prompt_grows_the_cache(self) -> None:
        """A call that needs more than the initial capacity reserves it up front."""
        target_model = _llama(2, 40)
        engine = _engine(target_model, [_perturbed(40, 0.003)], temperature=0.0, cache_len=16)
        result = engine.generate(_prompt(7, 30))
        assert result.n_tokens_generated == 40
        assert engine._target._inc._max_len >= 30 + 40 + 4  # type: ignore[attr-defined]

    def test_debug_checks_with_two_arms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every step's cache assertion holds, including the idle arm's catch-up."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        engine = _engine(
            _llama(3, 50),
            [_perturbed(50, 0.003), _perturbed(51, 0.02)],
            temperature=1.0,
            gamma=3,
            seed=5,
        )
        assert engine.generate(_prompt(8, 9)).n_tokens_generated == 40


def test_compile_flag_is_a_no_op_on_cpu(caplog: pytest.LogCaptureFixture) -> None:
    """compile=True on CPU warns and leaves the dynamic path in place."""
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(1, 60))
    drafter = HuggingFaceDrafter("d", device="cpu", model=_llama(1, 61))
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=8,
        compile=True,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=2),
    )
    with caplog.at_level(logging.WARNING, logger="flashspec.engine.speculative"):
        engine = SpeculativeEngine(
            config,
            [drafter],  # type: ignore[list-item]
            target,
            UCB1Selector(n_arms=1, window_size=0),
        )
    assert not engine.compiled
    assert "needs a CUDA device" in caplog.text
    assert isinstance(target._inc, KvSession)
    assert engine.generate(_prompt(9, 5)).n_tokens_generated == 8
