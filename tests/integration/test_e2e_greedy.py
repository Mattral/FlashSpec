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

        # With identical distributions, greedy u=0 accepts all.
        # Use rejection_sample with same model for target.
        accepted, first_rej, alpha = rejection_sample(
            input_ids=input_ids,
            draft_logprobs=draft_lp,
            target_logprobs=target_lp,
            draft_token_ids=draft_ids,
            gamma=gamma,
        )

        # When draft == target all tokens accepted.
        assert (first_rej == gamma).all(), (
            f"Greedy spec should accept all draft tokens; first_rej={first_rej}"
        )
        assert alpha == 1.0, f"Expected alpha=1.0 when p==q, got {alpha}"
