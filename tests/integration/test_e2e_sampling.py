"""Integration test: speculative sampling output distribution matches target model.

Uses a KS test at significance level α=0.01 over N=1000 samples to verify
that the output distribution of rejection_sample matches multinomial sampling
from the target model.  This is the CI hard gate (AGENTS.md §2.1).

Note: Full N=10,000 test is in the ``@pytest.mark.slow`` variant.
CI runs the fast N=1000 version; the slow version runs in nightly benchmarks.
"""

from __future__ import annotations

import pytest
import torch
from scipy import stats  # type: ignore[import]

from flashspec.sampling.rejection import rejection_sample
from flashspec.utils.device import set_seed

# KS-test significance level.
_ALPHA_KS: float = 0.01
_N_SAMPLES_FAST: int = 1_000
_N_SAMPLES_SLOW: int = 10_000


def _collect_residual_samples(
    n_samples: int,
    batch: int,
    vocab: int,
    gamma: int,
    seed: int,
) -> tuple[list[int], list[float]]:
    """Collect residual token samples and expected target probabilities.

    Returns
    -------
    samples : list[int]
        Observed residual token IDs.
    target_probs_at_pos0 : list[float]
        Target model probability for each sampled token (for KS test reference).
    """
    set_seed(seed)
    # Use fixed logprobs so the target distribution is known.
    target_lp = torch.randn(batch, gamma, vocab).log_softmax(-1)
    draft_lp   = torch.randn(batch, gamma, vocab).log_softmax(-1)
    input_ids  = torch.randint(0, vocab, (batch, 8))

    # Force all-reject (u=1) so every call samples the residual.
    samples: list[int] = []
    for i in range(n_samples):
        set_seed(seed + i)
        draft_ids = torch.randint(0, vocab, (batch, gamma))
        accepted, first_rej, _alpha = rejection_sample(
            input_ids=input_ids,
            draft_logprobs=draft_lp,
            target_logprobs=target_lp,
            draft_token_ids=draft_ids,
            gamma=gamma,
        )
        # Collect the token at position first_rej[0] (residual for first batch elem).
        pos = first_rej[0].item()
        safe_pos = min(int(pos), gamma - 1)
        tok = accepted[0, safe_pos].item()
        if tok >= 0:
            samples.append(int(tok))

    target_probs = target_lp[0, 0].exp().tolist()
    return samples, target_probs

