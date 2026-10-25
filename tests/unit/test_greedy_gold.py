"""Greedy FlashSpec equals ``model.generate(do_sample=False)``, token for token.

Greedy speculative decoding is lossless by construction: a draft token
is kept only where it equals the target's argmax, and every emitted
token is a target argmax. So the output must match the target's own
greedy decode exactly, for any drafter and any gamma. This file checks
that on a tiny random Llama for 5 prompts x 64 tokens and
gamma in {1, 2, 4, 6}, with three drafters: an unrelated model (nearly
every draft rejected), a slightly perturbed copy of the target (a mix
of partial and full accepts), and the target's own weights (every step
a full accept plus bonus).

``tests/integration/test_greedy_gold_gpu.py`` runs the same check on
TinyLlama-1.1B with a 160M drafter on CUDA.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import ForwardRecorder
from flashspec.sampling.greedy import greedy_verify
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

_VOCAB = 512
_N_PROMPTS = 5
_NEW_TOKENS = 64
_GAMMAS = (1, 2, 4, 6)


def _llama(layers: int, seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        initializer_range=0.1,
    )
    return LlamaForCausalLM(config).eval()


def _drafter_model(kind: str) -> LlamaForCausalLM:
    if kind == "unrelated":
        return _llama(2, 1)
    model = _llama(4, 0)
    if kind == "perturbed":
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.003)
    return model


def _prompts() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    return [
        torch.randint(0, _VOCAB, (1, length), generator=generator)
        for length in (1, 5, 9, 16, 23)[:_N_PROMPTS]
    ]


def _hf_greedy(model: LlamaForCausalLM, prompt: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=_NEW_TOKENS,
            min_new_tokens=_NEW_TOKENS,
            eos_token_id=None,
            pad_token_id=0,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return out[:, prompt.shape[1] :]


def _engine(
    drafters: list[HuggingFaceDrafter], target: TargetModel, gamma: int, profile: bool = False
) -> SpeculativeEngine:
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=_NEW_TOKENS,
        bandit=BanditConfig(n_arms=len(drafters), strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=0.0),
        metrics=MetricsConfig(profile_steps=profile),
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
    """Build the target shared by every gold case."""
    return _llama(4, 0)


@pytest.fixture(scope="module")
def expected(target_model: LlamaForCausalLM) -> list[torch.Tensor]:
    """HuggingFace greedy decode of each prompt: the reference."""
    return [_hf_greedy(target_model, prompt) for prompt in _prompts()]


@pytest.mark.parametrize("gamma", _GAMMAS)
@pytest.mark.parametrize("drafter_kind", ["unrelated", "perturbed", "same_weights"])
def test_greedy_matches_hf_generate(
    target_model: LlamaForCausalLM,
    expected: list[torch.Tensor],
    gamma: int,
    drafter_kind: str,
) -> None:
    """Every prompt, every gamma: identical tokens to generate(do_sample=False)."""
    drafter = HuggingFaceDrafter("d", device="cpu", model=_drafter_model(drafter_kind))
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
    engine = _engine([drafter], target, gamma, profile=True)
    emitted: list[int] = []
    for prompt, reference in zip(_prompts(), expected, strict=True):
        result = engine.generate(prompt)
        assert result.output_ids.shape == (1, _NEW_TOKENS)
        mismatch = (result.output_ids != reference).nonzero()
        assert mismatch.numel() == 0, (
            f"gamma={gamma}, drafter={drafter_kind}, prompt_len={prompt.shape[1]}: "
            f"first divergence at {mismatch[0].tolist()}"
        )
        assert result.step_profile is not None
        emitted += [step.tokens_emitted for step in result.step_profile]
    alpha = engine._acceptance_tracker.mean_acceptance_rate
    if drafter_kind == "same_weights":
        # Same weights: every draft token equals the target argmax.
        assert alpha == 1.0
    elif drafter_kind == "perturbed":
        # Both cache paths ran: a full accept with bonus, and a rejection.
        assert gamma + 1 in emitted, emitted
        assert any(count <= gamma for count in emitted[:-1]), emitted
        assert 0.0 < alpha < 1.0


def test_greedy_batch_of_two_matches_hf_generate(
    target_model: LlamaForCausalLM, expected: list[torch.Tensor]
) -> None:
    """The context-based batch > 1 loop is greedy-exact too, row by row."""
    prompts = _prompts()
    batch = torch.cat([prompts[2], prompts[2].flip(1)], dim=0)
    reference = torch.cat([expected[2], _hf_greedy(target_model, prompts[2].flip(1))], dim=0)
    drafter = HuggingFaceDrafter("d", device="cpu", model=_drafter_model("perturbed"))
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
    result = _engine([drafter], target, 3).generate(batch)
    assert torch.equal(result.output_ids, reference)


def test_greedy_two_drafter_arms_match_hf_generate(
    target_model: LlamaForCausalLM, expected: list[torch.Tensor]
) -> None:
    """Idle-arm catch-up keeps the greedy output exact."""
    drafters = [
        HuggingFaceDrafter("a", device="cpu", model=_drafter_model("perturbed")),
        HuggingFaceDrafter("b", device="cpu", model=_drafter_model("unrelated")),
    ]
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
    engine = _engine(drafters, target, 4)
    for prompt, reference in zip(_prompts(), expected, strict=True):
        assert torch.equal(engine.generate(prompt).output_ids, reference)


def test_greedy_uncached_models_match_argmax_decode() -> None:
    """Modules without a KV cache take the context-based greedy path."""

    class _NoCache(nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            torch.manual_seed(seed)
            self.embed = nn.Embedding(_VOCAB, 16)
            self.mix = nn.Linear(16, 16)
            self.head = nn.Linear(16, _VOCAB)
            self.vocab_size = _VOCAB

        def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
            hidden = self.embed(input_ids).cumsum(dim=1)
            return SimpleNamespace(logits=self.head(torch.tanh(self.mix(hidden))))

    target_module = _NoCache(3)
    context = torch.randint(0, _VOCAB, (1, 4))
    reference = []
    with torch.no_grad():
        running = context
        for _ in range(_NEW_TOKENS):
            next_id = target_module(running).logits[:, -1].argmax(-1, keepdim=True)
            reference.append(next_id)
            running = torch.cat([running, next_id], dim=1)
    drafter = HuggingFaceDrafter("d", device="cpu", temperature=0.0, model=_NoCache(4))
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_module)
    result = _engine([drafter], target, 4).generate(context)
    assert torch.equal(result.output_ids, torch.cat(reference, dim=1))


def test_greedy_step_is_one_target_forward_and_builds_no_softmax(
    target_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Greedy keeps the Phase 1 forward pattern and never calls (log_)softmax."""
    calls: list[str] = []
    for name in ("softmax", "log_softmax"):
        original = getattr(torch, name)

        def _counting(
            *args: object, _name: str = name, _f: object = original, **kw: object
        ) -> object:
            calls.append(_name)
            return _f(*args, **kw)  # type: ignore[operator]

        monkeypatch.setattr(torch, name, _counting)
    drafter_model = _drafter_model("perturbed")
    drafter = HuggingFaceDrafter("d", device="cpu", model=drafter_model)
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=target_model)
    gamma = 4
    engine = _engine([drafter], target, gamma, profile=True)
    prompt = _prompts()[3]
    with ForwardRecorder(target_model) as t_rec, ForwardRecorder(drafter_model) as d_rec:
        result = engine.generate(prompt)
    assert calls == []
    assert result.step_profile is not None
    n_steps = len(result.step_profile)
    assert t_rec.lengths == [prompt.shape[1], gamma] + [1 + gamma] * (n_steps - 1)
    assert len(d_rec.lengths) == 1 + gamma * n_steps


def test_greedy_draft_returns_no_logprobs() -> None:
    """The incremental greedy draft builds no full-vocab log-probs."""
    drafter = HuggingFaceDrafter("d", device="cpu", model=_llama(1, 5))
    assert drafter.begin(torch.randint(0, _VOCAB, (1, 4)))
    ids, logprobs = drafter.draft(3, greedy=True)
    assert ids.shape == (1, 3)
    assert logprobs is None


class TestGreedyVerify:
    """The acceptance rule on hand-made rows."""

    def test_mismatch_full_accept_and_bonus(self) -> None:
        """First mismatch takes the target token; a full accept exposes the bonus."""
        draft = torch.tensor([[1, 2, 3], [1, 2, 3], [7, 2, 3]])
        target = torch.tensor([[1, 9, 3, 4], [1, 2, 3, 4], [8, 2, 3, 4]])
        accepted, first, bonus = greedy_verify(draft, target)
        assert accepted.tolist() == [[1, 9, -1], [1, 2, 3], [8, -1, -1]]
        assert first.tolist() == [1, 3, 0]
        assert bonus is not None
        assert bonus.tolist() == [4, 4, 4]

    def test_without_bonus_column(self) -> None:
        """Only gamma columns: no bonus."""
        _, first, bonus = greedy_verify(torch.tensor([[5, 6]]), torch.tensor([[5, 6]]))
        assert first.tolist() == [2]
        assert bonus is None

    def test_shape_errors(self) -> None:
        """Mismatched shapes raise."""
        with pytest.raises(ValueError, match="must be"):
            greedy_verify(torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3, 4]]))
        with pytest.raises(ValueError, match="2-D"):
            greedy_verify(torch.tensor([1, 2]), torch.tensor([[1, 2]]))
