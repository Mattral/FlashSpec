"""Prompt-lookup (n-gram) drafter: lookup rule, greedy exactness, host reads, step functions.

The output-distribution checks for sampling with a one-hot ``q`` are in
``tests/integration/test_output_distribution_ngram.py``.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter, is_lookup_drafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import ForwardRecorder, count_host_syncs
from flashspec.sampling.greedy import greedy_step
from flashspec.sampling.rejection import rejection_step_onehot
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

_VOCAB = 256
_NEW = 48


def _reference(seq: list[int], gamma: int, sizes: tuple[int, ...] = (3, 2, 1)) -> list[int]:
    """Plain-Python lookup: the rule the device code must follow."""
    length = len(seq)
    for n in sizes:
        if length <= n:
            continue
        key = seq[length - n :]
        for start in range(length - n - 1, -1, -1):
            if seq[start : start + n] == key:
                cont = start + n
                return seq[cont : min(cont + gamma, length)]
    return []


class TestLookupRule:
    """``propose`` against a plain-Python reference."""

    def test_examples(self) -> None:
        """Longest key first, most recent occurrence, continuation clipped to the context."""
        drafter = NgramDrafter()
        buf = torch.zeros((1, 20), dtype=torch.long)
        seq = [5, 6, 7, 8, 9, 5, 6, 7]
        buf[0, : len(seq)] = torch.tensor(seq)
        ids, n = drafter.propose(buf, len(seq), 4)
        assert ids.tolist() == [[8, 9, 5, 6]] and n.tolist() == [4]
        ids, n = drafter.propose(buf, 3, 4)
        assert n.tolist() == [0]
        # Filler is the last context token, never a stale buffer value.
        assert ids.tolist() == [[7, 7, 7, 7]]
        ids, n = drafter.propose(torch.tensor([[1, 2, 1, 2, 0, 0]]), 4, 4)
        assert ids.tolist() == [[1, 2, 2, 2]] and n.tolist() == [2]

    @pytest.mark.parametrize("vocab", [3, 8, 50])
    def test_matches_reference_on_random_sequences(self, vocab: int) -> None:
        """Random contexts, random lengths, stale tails, int and tensor lengths."""
        rng = random.Random(vocab)
        drafter = NgramDrafter()
        for _ in range(300):
            length = rng.randint(1, 40)
            gamma = rng.randint(1, 7)
            seq = [rng.randrange(vocab) for _ in range(length)]
            stale = [rng.randrange(vocab) for _ in range(rng.randint(0, 10))]
            buf = torch.tensor([seq + stale], dtype=torch.long)
            want = _reference(seq, gamma)
            for as_tensor in (False, True):
                arg = torch.tensor(length) if as_tensor else length
                ids, n = drafter.propose(buf, arg, gamma)
                assert ids.shape == (1, gamma)
                assert int(n) == len(want), (seq, gamma)
                assert ids[0, : len(want)].tolist() == want, (seq, gamma)
                assert bool((ids >= 0).all()) and bool((ids < vocab).all())

    def test_sizes_and_validation(self) -> None:
        """Only the configured key lengths are tried; bad arguments raise."""
        assert NgramDrafter().ngram_sizes == (3, 2, 1)
        two = NgramDrafter(max_ngram=2, min_ngram=2)
        # Only a one-token key would match here.
        assert int(two.propose(torch.tensor([[4, 9, 3, 4]]), 4, 2)[1]) == 0
        assert int(NgramDrafter().propose(torch.tensor([[4, 9, 3, 4]]), 4, 2)[1]) == 2
        with pytest.raises(ValueError, match="min_ngram"):
            NgramDrafter(max_ngram=1, min_ngram=2)
        with pytest.raises(ValueError, match="gamma"):
            NgramDrafter().propose(torch.zeros((1, 4), dtype=torch.long), 2, 0)
        with pytest.raises(ValueError, match="capacity"):
            NgramDrafter().propose(torch.zeros((2, 4), dtype=torch.long), 2, 1)
        assert is_lookup_drafter(NgramDrafter())
        assert not is_lookup_drafter(object())

    def test_reads_nothing_from_the_device(self) -> None:
        """No tolist, item or bool inside propose (CPU proxy counter)."""
        buf = torch.randint(0, 5, (1, 64))
        with count_host_syncs("cpu") as counter:
            NgramDrafter().propose(buf, torch.tensor(40), 6)
        assert counter.count == 0, counter.by_kind


class TestStepFunctions:
    """``greedy_step`` with a proposal length and the one-hot Algorithm 1 step."""

    def test_greedy_step_stops_at_draft_len(self) -> None:
        """Matching tokens past the real length are not accepted."""
        draft = torch.tensor([[3, 4, 5, 6]])
        argmax = torch.tensor([[3, 4, 5, 6, 7]])
        assert greedy_step(draft, argmax).tolist() == [[4, 7]]
        assert greedy_step(draft, argmax, torch.tensor([2])).tolist() == [[2, 5]]
        assert greedy_step(draft, argmax, torch.tensor([0])).tolist() == [[0, 3]]
        mismatch = torch.tensor([[3, 9, 5, 6, 7]])
        assert greedy_step(draft, mismatch, torch.tensor([3])).tolist() == [[1, 9]]

    def test_onehot_certain_and_impossible_tokens(self) -> None:
        """p(x) = 1 always accepts; p(x) = 0 always rejects and never re-emits x."""
        vocab = 5
        sure = torch.full((1, 3, vocab), -1e9)
        sure[0, :, 2] = 0.0
        step = rejection_step_onehot(torch.tensor([[2, 2]]), torch.log_softmax(sure, -1))
        assert step.tolist() == [[2, 2]]
        logits = torch.zeros((200, 3, vocab))
        logits[:, :, 1] = -1e9  # p(1) = 0
        out = rejection_step_onehot(
            torch.ones((200, 2), dtype=torch.long), torch.log_softmax(logits, -1)
        )
        assert bool((out[:, 0] == 0).all())
        assert bool((out[:, 1] != 1).all())

    def test_onehot_draft_len_zero_and_no_bonus_row(self) -> None:
        """draft_len 0 samples the first row; a full accept without a bonus row returns -1."""
        vocab = 4
        logits = torch.full((1, 3, vocab), -1e9)
        logits[0, 0, 3] = 0.0  # row 0 is certain about token 3
        logits[0, 1:, 1] = 0.0
        step = rejection_step_onehot(
            torch.tensor([[1, 1]]), torch.log_softmax(logits, -1), torch.tensor([0])
        )
        assert step.tolist() == [[0, 3]]
        two_rows = torch.log_softmax(torch.zeros((1, 2, vocab)), -1)
        sure = torch.full((1, 2, vocab), -1e9)
        sure[0, :, 0] = 0.0
        full = rejection_step_onehot(torch.tensor([[0, 0]]), torch.log_softmax(sure, -1))
        assert full.tolist() == [[2, -1]]
        with pytest.raises(ValueError, match="target_logprobs"):
            rejection_step_onehot(torch.tensor([[0, 0]]), two_rows[:, :1])


# ── Engine: greedy exactness with prompt lookup ──────────────────────────────


def _llama(layers: int, seed: int) -> LlamaForCausalLM:
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
    return LlamaForCausalLM(config).eval()


def _perturbed(seed: int, scale: float) -> LlamaForCausalLM:
    model = _llama(3, seed)
    generator = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * scale)
    return model


def _prompts() -> list[torch.Tensor]:
    """Prompts with repeated spans, so lookup finds matches from the start."""
    generator = torch.Generator().manual_seed(77)
    out = []
    for length in (1, 6, 12):
        chunk = torch.randint(0, _VOCAB, (1, length), generator=generator)
        out.append(torch.cat([chunk, chunk[:, : max(1, length // 2)]], dim=1))
    out.append(torch.randint(0, _VOCAB, (1, 9), generator=generator))
    return out


def _hf_greedy(model: LlamaForCausalLM, prompt: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=_NEW,
            min_new_tokens=_NEW,
            eos_token_id=None,
            pad_token_id=0,
        )
    return out[:, prompt.shape[1] :]


def _engine(
    target_model: LlamaForCausalLM,
    drafters: list[object],
    *,
    gamma: int,
    temperature: float = 0.0,
    static: bool = False,
    seed: int | None = None,
) -> SpeculativeEngine:
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
    if static:
        for owner in (target, *drafters):
            use = getattr(owner, "use_static_cache", None)
            if callable(use):
                use(64, compile_forward=False)
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=_NEW,
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


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


@pytest.fixture(scope="module")
def target_model() -> LlamaForCausalLM:
    """Target shared by the greedy cases."""
    return _llama(3, 10)


@pytest.fixture(scope="module")
def expected(target_model: LlamaForCausalLM) -> list[torch.Tensor]:
    """HuggingFace greedy decode of each prompt."""
    return [_hf_greedy(target_model, prompt) for prompt in _prompts()]


class TestGreedyExact:
    """Greedy FlashSpec with a lookup arm equals generate(do_sample=False)."""

    @pytest.mark.parametrize("static", [False, True])
    @pytest.mark.parametrize("gamma", [1, 2, 4, 6])
    def test_ngram_only(
        self,
        target_model: LlamaForCausalLM,
        expected: list[torch.Tensor],
        gamma: int,
        static: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lookup-only engine: exact tokens, and lookup both hits and misses."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        engine = _engine(target_model, [NgramDrafter()], gamma=gamma, static=static)
        emitted: list[int] = []
        for prompt, reference in zip(_prompts(), expected, strict=True):
            result = engine.generate(prompt)
            assert torch.equal(result.output_ids, reference), (gamma, static, prompt.shape)
            assert result.step_profile is not None
            emitted += [step.tokens_emitted for step in result.step_profile]
        # Some steps accepted lookup tokens, and some were plain decode steps.
        assert max(emitted) > 1 and min(emitted) == 1, emitted

    @pytest.mark.parametrize("static", [False, True])
    def test_ngram_and_model_arms(
        self,
        target_model: LlamaForCausalLM,
        expected: list[torch.Tensor],
        static: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two arms: the model drafter catches up after lookup steps; output stays exact."""
        monkeypatch.setenv("FLASHSPEC_DEBUG_CACHE", "1")
        drafter = HuggingFaceDrafter("d", device="cpu", model=_perturbed(10, 0.003))
        engine = _engine(target_model, [NgramDrafter(), drafter], gamma=4, static=static)
        for prompt, reference in zip(_prompts(), expected, strict=True):
            assert torch.equal(engine.generate(prompt).output_ids, reference)
        pulls = [arm.n_pulls for arm in engine._bandit._arms]
        assert all(count > 0 for count in pulls), pulls

    def test_one_target_forward_per_step(self, target_model: LlamaForCausalLM) -> None:
        """A lookup step is one target forward of pending + gamma tokens; no drafter model runs."""
        engine = _engine(target_model, [NgramDrafter()], gamma=4)
        prompt = _prompts()[2]
        with ForwardRecorder(target_model) as recorder:
            result = engine.generate(prompt)
        assert result.step_profile is not None
        n_steps = len(result.step_profile)
        assert recorder.lengths == [int(prompt.shape[1]), 4] + [5] * (n_steps - 1)

    @pytest.mark.parametrize("temperature", [0.0, 0.7])
    def test_one_host_read_per_step(
        self, target_model: LlamaForCausalLM, temperature: float
    ) -> None:
        """The lookup arm keeps the single [n, token, drafts] read per step."""
        engine = _engine(target_model, [NgramDrafter()], gamma=4, temperature=temperature, seed=3)
        with count_host_syncs("cpu") as counter:
            result = engine.generate(_prompts()[2])
        assert result.step_profile is not None
        assert counter.count == len(result.step_profile), counter.by_kind

    def test_sampling_is_seeded_and_runs(self, target_model: LlamaForCausalLM) -> None:
        """Sampling with a lookup arm: same seed, same tokens; all ids in the vocabulary."""
        first = _engine(target_model, [NgramDrafter()], gamma=4, temperature=1.0, seed=9)
        second = _engine(target_model, [NgramDrafter()], gamma=4, temperature=1.0, seed=9)
        prompt = _prompts()[1]
        a, b = first.generate(prompt).output_ids, second.generate(prompt).output_ids
        assert torch.equal(a, b)
        assert bool((a >= 0).all()) and bool((a < _VOCAB).all())

    def test_generic_loop_refuses_lookup(self, target_model: LlamaForCausalLM) -> None:
        """Batch > 1 has no lookup path; it says so instead of producing wrong output."""
        engine = _engine(target_model, [NgramDrafter()], gamma=2)
        with pytest.raises(ValueError, match="batch-1 fast path"):
            engine.generate(torch.zeros((2, 4), dtype=torch.long))
