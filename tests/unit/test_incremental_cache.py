"""Incremental KV bookkeeping: integer crops and pending tokens, no catch-up forwards.

The engine tells each wrapper how many draft tokens were accepted and
which token comes next. These tests check that the wrappers' caches then
hold exactly the accepted sequence (``FLASHSPEC_DEBUG_CACHE=1`` makes
the engine assert that after every step), that the logits match a
cache-free forward, and that failures fall back to the context-based
methods.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.kv_cache import CacheUnavailableError, KvSession
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import ForwardRecorder
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig

_VOCAB = 128


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


def _full_logprobs(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.log_softmax(model(token_ids).logits.float(), dim=-1)


def _engine(
    drafters: list[HuggingFaceDrafter], target: TargetModel, gamma: int, max_new: int
) -> SpeculativeEngine:
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=max_new,
        bandit=BanditConfig(n_arms=len(drafters), strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma),
    )
    return SpeculativeEngine(
        config,
        drafters,  # type: ignore[arg-type]
        target,
        UCB1Selector(n_arms=len(drafters), window_size=0),
    )


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


class TestDebugCacheAssertions:
    """With FLASHSPEC_DEBUG_CACHE=1 the engine checks every cache after every step."""

    @pytest.mark.parametrize(("gamma", "seed"), [(4, 0), (1, 1), (3, 2)])
    def test_single_drafter_caches_match_context(
        self, monkeypatch: pytest.MonkeyPatch, gamma: int, seed: int
    ) -> None:
        """Rejections and full accepts both leave cache + pending == accepted context."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        drafter = HuggingFaceDrafter("d", device="cpu", model=_llama(1, seed + 1))
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(2, seed))
        checks: list[int] = []
        original = target.check_cache

        def _spy(context: torch.Tensor) -> None:
            checks.append(int(context.shape[1]))
            original(context)

        monkeypatch.setattr(target, "check_cache", _spy)
        prompt = torch.randint(0, _VOCAB, (1, 6))
        result = _engine([drafter], target, gamma, 30).generate(prompt)
        assert result.n_tokens_generated == 30
        assert checks, "the debug path never ran"

    def test_two_drafters_idle_arm_catches_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The arm the bandit skips queues the step's tokens and stays aligned."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        drafter_models = [_llama(1, 11), _llama(1, 12)]
        drafters = [
            HuggingFaceDrafter(f"d{index}", device="cpu", model=model)
            for index, model in enumerate(drafter_models)
        ]
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(2, 10))
        engine = _engine(drafters, target, 3, 40)
        prompt = torch.randint(0, _VOCAB, (1, 5))
        recorders = [ForwardRecorder(model).attach() for model in drafter_models]
        try:
            result = engine.generate(prompt)
        finally:
            for recorder in recorders:
                recorder.detach()
        assert result.n_tokens_generated == 40
        # UCB1 plays every arm, so both drafters started and drafted. The
        # debug checks after every step are the alignment assertion.
        assert all(recorder.calls > 0 for recorder in recorders)
        # Each drafter forwards every committed token once at most: no
        # re-forward of the prompt or of tokens it already holds.
        for recorder in recorders:
            assert sum(recorder.lengths) <= 5 + 40 + 3 * recorder.calls

    def test_check_raises_on_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The assertion path really compares ids."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        session = KvSession(_llama(1, 3))
        prompt = torch.randint(0, _VOCAB, (1, 5))
        session.start(prompt)
        session.check(prompt)
        with pytest.raises(AssertionError, match="do not match"):
            session.check((prompt + 1) % _VOCAB)

    def test_check_is_a_no_op_without_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off by default: nothing is compared."""
        monkeypatch.delenv("FLASHSPEC_DEBUG_CACHE", raising=False)
        session = KvSession(_llama(1, 3))
        prompt = torch.randint(0, _VOCAB, (1, 5))
        session.start(prompt)
        session.check((prompt + 1) % _VOCAB)


class TestLogitsMatchFullForward:
    """Incremental rows are the rows a cache-free forward would produce."""

    def test_verify_commit_verify(self) -> None:
        """Second verify after a partial accept scores like a fresh full forward."""
        torch.manual_seed(0)
        model = _llama(2, 4)
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=model)
        prompt = torch.randint(0, _VOCAB, (1, 7))
        draft = torch.randint(0, _VOCAB, (1, 4))
        assert target.begin(prompt)
        first = target.verify(draft, temperature=1.0)
        full = _full_logprobs(model, torch.cat([prompt, draft], dim=1))
        assert torch.allclose(first, full[:, 6:11], rtol=1e-4, atol=1e-5)

        residual = torch.tensor([[5]])
        target.commit(2, residual)
        accepted = torch.cat([prompt, draft[:, :2], residual], dim=1)
        second_draft = torch.randint(0, _VOCAB, (1, 4))
        with ForwardRecorder(model) as recorder:
            second = target.verify(second_draft, temperature=0.8)
        assert recorder.lengths == [1 + 4]
        with torch.no_grad():
            logits = model(torch.cat([accepted, second_draft], dim=1)).logits.float() / 0.8
        expected = torch.log_softmax(logits, dim=-1)[:, accepted.shape[1] - 1 :]
        assert torch.allclose(second, expected, rtol=1e-4, atol=1e-5)

    def test_drafter_full_accept_then_draft(self) -> None:
        """After a full accept the next draft forwards [last draft, bonus] and matches."""
        model = _llama(1, 5)
        drafter = HuggingFaceDrafter("d", device="cpu", model=model)
        prompt = torch.randint(0, _VOCAB, (1, 6))
        assert drafter.begin(prompt)
        torch.manual_seed(1)
        ids, _ = drafter.draft(3)
        bonus = torch.tensor([[9]])
        drafter.commit(3, bonus)
        accepted = torch.cat([prompt, ids, bonus], dim=1)
        torch.manual_seed(2)
        with ForwardRecorder(model) as recorder:
            next_ids, next_lp = drafter.draft(2)
        assert recorder.lengths == [2, 1]
        full = _full_logprobs(model, torch.cat([accepted, next_ids], dim=1))
        expected = full[:, accepted.shape[1] - 1 : accepted.shape[1] + 1]
        assert torch.allclose(next_lp, expected, rtol=1e-4, atol=1e-5)

    def test_idle_drafter_push_pending(self) -> None:
        """Tokens another arm committed are forwarded once, in front of the next draft."""
        model = _llama(1, 6)
        drafter = HuggingFaceDrafter("d", device="cpu", model=model)
        prompt = torch.randint(0, _VOCAB, (1, 4))
        assert drafter.begin(prompt)
        elsewhere = torch.randint(0, _VOCAB, (1, 3))
        drafter.push_pending(elsewhere)
        with ForwardRecorder(model) as recorder:
            ids, logprobs = drafter.draft(1)
        assert recorder.lengths == [1 + 3]
        context = torch.cat([prompt, elsewhere], dim=1)
        expected = _full_logprobs(model, context)[:, -1:]
        assert torch.allclose(logprobs, expected, rtol=1e-4, atol=1e-5)
        assert ids.shape == (1, 1)


class TestPrefill:
    """The prompt is prefilled once, without materialising its logits."""

    def test_prefill_keeps_one_logit_row(self) -> None:
        """The prefill forward asks for one logit row."""
        model = _llama(1, 7)
        seen: list[object] = []

        def _hook(_m: nn.Module, _args: tuple[object, ...], kwargs: dict[str, object]) -> None:
            seen.append(kwargs.get("logits_to_keep"))

        handle = model.register_forward_pre_hook(_hook, with_kwargs=True)
        try:
            session = KvSession(model)
            session.start(torch.randint(0, _VOCAB, (1, 9)))
            logits = session.advance(torch.randint(0, _VOCAB, (1, 3)), keep=4)
        finally:
            handle.remove()
        assert seen == [1, 4]
        assert logits.shape == (1, 4, _VOCAB)
        assert session.cached_length == 12
        assert session.pending_length == 0

    def test_one_token_prompt(self) -> None:
        """A one-token prompt is prefilled like any other; the verify scores against it."""
        model = _llama(1, 8)
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=model)
        prompt = torch.tensor([[3]])
        with ForwardRecorder(model) as recorder:
            assert target.begin(prompt)
            scored = target.verify(torch.tensor([[4, 5]]))
        assert recorder.lengths == [1, 2]
        expected = _full_logprobs(model, torch.tensor([[3, 4, 5]]))
        assert torch.allclose(scored, expected, rtol=1e-4, atol=1e-5)

    def test_model_without_logits_to_keep_is_sliced(self) -> None:
        """A cache model that ignores ``logits_to_keep`` still returns ``keep`` rows."""

        class _NoKeep(nn.Module):
            def __init__(self, inner: LlamaForCausalLM) -> None:
                super().__init__()
                self.inner = inner

            def forward(
                self,
                input_ids: torch.Tensor,
                past_key_values: object = None,
                use_cache: bool = True,
            ) -> object:
                return self.inner(
                    input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache
                )

        session = KvSession(_NoKeep(_llama(1, 9)))
        session.start(torch.randint(0, _VOCAB, (1, 4)))
        assert session.advance(torch.randint(0, _VOCAB, (1, 2)), keep=2).shape == (1, 2, _VOCAB)


class TestFallbacksAndGuards:
    """Errors leave the engine on the context-based path; misuse raises."""

    def test_verify_failure_falls_back_to_score_draft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache failure mid-call switches the target to full-context scoring."""
        drafter = HuggingFaceDrafter("d", device="cpu", model=_llama(1, 21))
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(2, 20))
        calls = {"verify": 0, "score": 0}
        original_score = target.score_draft

        def _broken_verify(*_args: object, **_kwargs: object) -> torch.Tensor:
            calls["verify"] += 1
            target._kv.disable()
            raise CacheUnavailableError("simulated")

        def _score(*args: object, **kwargs: object) -> torch.Tensor:
            calls["score"] += 1
            return original_score(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(target, "verify", _broken_verify)
        monkeypatch.setattr(target, "score_draft", _score)
        result = _engine([drafter], target, 3, 12).generate(torch.randint(0, _VOCAB, (1, 5)))
        assert result.n_tokens_generated == 12
        assert calls["verify"] == 1
        assert calls["score"] >= 1

    def test_uncached_model_reports_no_incremental_path(self) -> None:
        """A model that does not accept past_key_values cannot begin."""

        class _NoCache(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(_VOCAB, 8)
                self.head = nn.Linear(8, _VOCAB)

            def forward(self, input_ids: torch.Tensor) -> object:
                from types import SimpleNamespace

                return SimpleNamespace(logits=self.head(self.embed(input_ids)))

        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_NoCache())
        drafter = HuggingFaceDrafter("d", device="cpu", model=_NoCache())
        prompt = torch.randint(0, _VOCAB, (1, 4))
        assert not target.begin(prompt)
        assert not drafter.begin(prompt)
        assert not target.begin(torch.randint(0, _VOCAB, (2, 4)))

    def test_commit_guards(self) -> None:
        """Commit rejects counts outside the last draft or verify."""
        target = TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(1, 30))
        drafter = HuggingFaceDrafter("d", device="cpu", model=_llama(1, 31))
        prompt = torch.randint(0, _VOCAB, (1, 4))
        assert target.begin(prompt)
        assert drafter.begin(prompt)
        with pytest.raises(ValueError, match="after draft"):
            drafter.commit(0, torch.tensor([[1]]))
        target.verify(torch.tensor([[1, 2]]))
        with pytest.raises(ValueError, match="n_accepted"):
            target.commit(3, torch.tensor([[1]]))
        with pytest.raises(ValueError, match="temperature"):
            target.verify(torch.tensor([[1]]), temperature=0.0)
        with pytest.raises(ValueError, match="gamma"):
            drafter.draft(0)

    def test_session_guards(self) -> None:
        """Incremental calls refuse impossible requests."""
        session = KvSession(_llama(1, 32))
        with pytest.raises(CacheUnavailableError):
            session.advance(None, keep=1)
        session.start(torch.randint(0, _VOCAB, (1, 3)))
        with pytest.raises(CacheUnavailableError, match="keep"):
            session.advance(None, keep=2)
        session.crop(0)
        with pytest.raises(CacheUnavailableError, match="cannot crop"):
            session.crop(50)
        assert not session.enabled
