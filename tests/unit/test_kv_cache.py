"""KV cache forwards only new tokens and matches a full forward.

The drafter, the verifier, and the autoregressive baseline used to rerun
the whole context on every token. These tests count the token lengths
passed into ``forward`` and compare log-probabilities with a cache-free
forward of the same tiny Llama.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.harness import autoregressive_generate
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig

_PROMPT_LEN = 8
_GAMMA = 4
_VOCAB = 32


def _llama() -> LlamaForCausalLM:
    """Build a tiny causal LM with a real ``DynamicCache``. No weights are downloaded."""
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _record(lengths: list[int], model: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Append each ``input_ids`` length seen by ``model.forward``."""

    def _hook(_module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        token_ids = kwargs.get("input_ids")
        if token_ids is None and args:
            token_ids = args[0]
        if isinstance(token_ids, torch.Tensor) and token_ids.ndim == 2:
            lengths.append(int(token_ids.shape[1]))

    return model.register_forward_pre_hook(_hook, with_kwargs=True)


def _full_logprobs(
    model: nn.Module, token_ids: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Log-softmax of one cache-free forward. Temperature is applied first."""
    with torch.no_grad():
        logits = model(token_ids).logits.float()
    if temperature != 1.0:
        logits = logits / temperature
    return torch.log_softmax(logits, dim=-1)


def test_drafter_second_step_forwards_only_new_tokens() -> None:
    """Once the draft is cached, the next step forwards one token at a time."""
    torch.manual_seed(0)
    model = _llama()
    drafter = HuggingFaceDrafter("tiny-llama", device="cpu", model=model)
    context = torch.randint(0, _VOCAB, (1, _PROMPT_LEN))
    draft_ids, draft_logprobs = drafter.generate_draft(context, gamma=_GAMMA)
    assert draft_ids.shape == (1, _GAMMA)
    assert draft_logprobs.shape == (1, _GAMMA, _VOCAB)

    lengths: list[int] = []
    hook = _record(lengths, model)
    try:
        extended = torch.cat([context, draft_ids], dim=1)
        again, _again_lp = drafter.generate_draft(extended, gamma=_GAMMA)
    finally:
        hook.remove()
    assert lengths == [1] * _GAMMA, lengths
    assert again.shape == (1, _GAMMA)


def test_drafter_rollback_logprobs_match_full_forward() -> None:
    """A diverging accepted prefix is cropped, then scored like a fresh forward."""
    torch.manual_seed(1)
    model = _llama()
    drafter = HuggingFaceDrafter("tiny-llama", device="cpu", temperature=1.3, model=model)
    context = torch.randint(0, _VOCAB, (1, _PROMPT_LEN))
    draft_ids, _draft_lp = drafter.generate_draft(context, gamma=_GAMMA)
    replacement = (draft_ids[:, 2:3] + 1) % _VOCAB
    accepted = torch.cat([context, draft_ids[:, :2], replacement], dim=1)
    fixed_draft = torch.randint(0, _VOCAB, (1, _GAMMA))

    lengths: list[int] = []
    hook = _record(lengths, model)
    try:
        scored = drafter.compute_logprobs(accepted, fixed_draft)
    finally:
        hook.remove()

    full = _full_logprobs(model, torch.cat([accepted, fixed_draft], dim=1), temperature=1.3)
    expected = full[:, accepted.shape[1] - 1 : accepted.shape[1] - 1 + _GAMMA, :]
    assert torch.allclose(scored, expected, rtol=1e-4, atol=1e-5)
    # Crop back to the divergence, replay the last kept token, then the draft.
    assert lengths == [1, _GAMMA], lengths


def test_verifier_cache_matches_full_forward_and_rolls_back() -> None:
    """Bonus rows match a full forward, and the next score does not repeat the prompt."""
    torch.manual_seed(2)
    model = _llama()
    target = TargetModel("tiny-llama", torch.device("cpu"), torch.float32, model=model)
    context = torch.randint(0, _VOCAB, (1, _PROMPT_LEN))
    draft = torch.randint(0, _VOCAB, (1, _GAMMA))
    scored = target.score_draft(context, draft, gamma=_GAMMA, temperature=1.7, include_bonus=True)
    full = _full_logprobs(model, torch.cat([context, draft], dim=1), temperature=1.7)
    expected = full[:, _PROMPT_LEN - 1 : _PROMPT_LEN + _GAMMA, :]
    assert scored.shape == (1, _GAMMA + 1, _VOCAB)
    assert torch.allclose(scored, expected, rtol=1e-4, atol=1e-5)

    other = torch.randint(0, _VOCAB, (1, _GAMMA))
    lengths: list[int] = []
    hook = _record(lengths, model)
    try:
        without_bonus = target.score_draft(context, other, gamma=_GAMMA, include_bonus=False)
    finally:
        hook.remove()
    full_other = _full_logprobs(model, torch.cat([context, other], dim=1))
    expected_other = full_other[:, _PROMPT_LEN - 1 : _PROMPT_LEN - 1 + _GAMMA, :]
    assert without_bonus.shape == (1, _GAMMA, _VOCAB)
    assert torch.allclose(without_bonus, expected_other, rtol=1e-4, atol=1e-5)
    assert lengths == [1, _GAMMA], lengths


def test_batch_two_stays_on_the_full_forward() -> None:
    """A single cache is not applied to two rows. Each step still sees the growing context."""
    torch.manual_seed(3)
    model = _llama()
    drafter = HuggingFaceDrafter("tiny-llama", device="cpu", model=model)
    context = torch.randint(0, _VOCAB, (2, _PROMPT_LEN))
    lengths: list[int] = []
    hook = _record(lengths, model)
    try:
        token_ids, logprobs = drafter.generate_draft(context, gamma=2)
    finally:
        hook.remove()
    assert token_ids.shape == (2, 2)
    assert logprobs.shape == (2, 2, _VOCAB)
    assert lengths == [_PROMPT_LEN, _PROMPT_LEN + 1], lengths


def test_autoregressive_baseline_uses_the_kv_cache() -> None:
    """The baseline prefill is the prompt. Every later forward is one token."""
    torch.manual_seed(4)
    model = _llama()
    context = torch.randint(0, _VOCAB, (1, _PROMPT_LEN))
    lengths: list[int] = []
    hook = _record(lengths, model)
    try:
        tokens = autoregressive_generate(model, context, max_new_tokens=_GAMMA, temperature=1.0)
    finally:
        hook.remove()
    assert tokens.shape == (1, _GAMMA)
    # Prefill reads the prompt. Later forwards are one token. HuggingFace
    # generate samples the first new token from the prefill, so there are
    # ``max_new_tokens - 1`` decode forwards after that prefill.
    assert lengths[0] == _PROMPT_LEN, lengths
    assert lengths[1:] == [1] * (_GAMMA - 1), lengths


def test_engine_steps_do_not_reforward_the_prompt() -> None:
    """Drafter and verifier keep the accepted prefix, including a later step."""
    torch.manual_seed(5)
    drafter_model = _llama()
    target_model = _llama()
    drafter = HuggingFaceDrafter("draft", device="cpu", model=drafter_model)
    target = TargetModel("target", torch.device("cpu"), torch.float32, model=target_model)
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=6,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=_GAMMA, temperature=1.0, seed=5),
    )
    engine = SpeculativeEngine(
        config,
        [drafter],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=1, window_size=0),
    )
    prompt = torch.randint(0, _VOCAB, (1, _PROMPT_LEN))
    lengths: list[int] = []
    drafter_hook = _record(lengths, drafter_model)
    target_hook = _record(lengths, target_model)
    try:
        result = engine.generate(prompt, max_new_tokens=6)
    finally:
        drafter_hook.remove()
        target_hook.remove()
    assert result.n_tokens_generated == 6
    # Each model prefills the prompt minus its last token once. After
    # that the target forwards 1 + gamma and the drafter 1 or 2 tokens.
    assert lengths.count(_PROMPT_LEN - 1) == 2, lengths
    assert max(lengths) == _PROMPT_LEN - 1, lengths
    assert 1 in lengths
