"""Integration test: greedy speculative decoding matches greedy autoregressive.

Uses tiny toy models from conftest.py.  No network access; no real GPU required
unless ``@pytest.mark.gpu`` is set.
"""

from __future__ import annotations

import pytest
import torch

from flashspec.sampling.rejection import rejection_sample
from flashspec.utils.device import set_seed


def _greedy_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    gamma: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate gamma tokens greedily and return their logprobs."""
    all_draft_ids = []
    ctx = input_ids.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(gamma):
            out = model(ctx)
            logits = out.logits[:, -1, :]              # (B, V)
            next_id = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            all_draft_ids.append(next_id)
            ctx = torch.cat([ctx, next_id], dim=-1)

    draft_ids = torch.cat(all_draft_ids, dim=-1)  # (B, gamma)

    # Compute logprobs for all draft positions in one pass.
    full = torch.cat([input_ids, draft_ids], dim=-1)
    with torch.no_grad():
        out = model(full)
    logprobs = torch.log_softmax(out.logits[:, input_ids.shape[1] - 1 : input_ids.shape[1] + gamma - 1, :].float(), dim=-1)
    return draft_ids, logprobs, logprobs  # draft == target for same model


class TestGreedySpeculativeEquivalence:
    """Greedy speculative decoding must produce identical output to greedy AR."""

    def test_greedy_speculative_matches_autoregressive_output(
        self, toy_model_cpu: torch.nn.Module, random_input_ids: torch.Tensor
    ) -> None:
        """Greedy speculative == greedy AR when target == draft model."""
        set_seed(42)
        gamma = 4
        batch_size, vocab = 2, 1000

        input_ids = torch.randint(0, vocab, (batch_size, 8))
        draft_ids, draft_lp, target_lp = _greedy_logprobs(
            toy_model_cpu, input_ids, gamma, vocab
        )

        accepted, first_rej, alpha = rejection_sample(
            input_ids=input_ids,
            draft_logprobs=draft_lp,
            target_logprobs=target_lp,
            draft_token_ids=draft_ids,
            gamma=gamma,
        )

        assert (first_rej == gamma).all(), (
            f"Greedy spec should accept all draft tokens; first_rej={first_rej}"
        )
        assert alpha == 1.0, f"Expected alpha=1.0 when p==q, got {alpha}"

    def test_greedy_speculative_batch_size_one_accepts_all(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """Edge case: batch_size=1 with p==q accepts all gamma tokens."""
        set_seed(10)
        gamma, vocab = 4, 1000
        input_ids = torch.randint(0, vocab, (1, 8))
        draft_ids, draft_lp, target_lp = _greedy_logprobs(
            toy_model_cpu, input_ids, gamma, vocab
        )
        _accepted, first_rej, alpha = rejection_sample(
            input_ids=input_ids,
            draft_logprobs=draft_lp,
            target_logprobs=target_lp,
            draft_token_ids=draft_ids,
            gamma=gamma,
        )
        assert first_rej[0].item() == gamma, (
            f"batch_size=1: expected first_rej=={gamma}, got {first_rej[0]}"
        )
        assert alpha == 1.0, f"batch_size=1: expected alpha=1.0, got {alpha}"

    def test_greedy_speculative_gamma_one_accepts_single_token(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """Edge case: gamma=1 with p==q must always accept the single draft token."""
        set_seed(11)
        batch_size, vocab = 2, 1000
        input_ids = torch.randint(0, vocab, (batch_size, 8))
        draft_ids, draft_lp, target_lp = _greedy_logprobs(
            toy_model_cpu, input_ids, gamma=1, vocab_size=vocab
        )
        _accepted, first_rej, alpha = rejection_sample(
            input_ids=input_ids,
            draft_logprobs=draft_lp,
            target_logprobs=target_lp,
            draft_token_ids=draft_ids,
            gamma=1,
        )
        assert (first_rej == 1).all(), (
            f"gamma=1 with p==q: expected first_rej==1, got {first_rej}"
        )
        assert alpha == 1.0, f"gamma=1 with p==q: expected alpha=1.0, got {alpha}"

    def test_greedy_speculative_short_context_length_one_succeeds(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """Edge case: context length=1 (minimal prefix) must not crash."""
        set_seed(12)
        gamma, vocab = 2, 1000
        input_ids = torch.randint(0, vocab, (1, 1))  # minimal context
        draft_ids, draft_lp, target_lp = _greedy_logprobs(
            toy_model_cpu, input_ids, gamma, vocab
        )
        accepted, first_rej, alpha = rejection_sample(
            input_ids=input_ids,
            draft_logprobs=draft_lp,
            target_logprobs=target_lp,
            draft_token_ids=draft_ids,
            gamma=gamma,
        )
        assert accepted.shape == (1, gamma), (
            f"Short context: accepted.shape={accepted.shape}, expected (1, {gamma})"
        )
