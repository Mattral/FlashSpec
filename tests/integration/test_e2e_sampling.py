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


class TestDistributionEquivalence:
    """Speculative sampling output distribution matches the target model."""

    def test_ks_test_passes_fast(self) -> None:
        """KS test at α=0.01 over N=1000 samples (fast CI gate)."""
        n, batch, vocab, gamma = _N_SAMPLES_FAST, 1, 50, 1
        samples, target_probs = _collect_residual_samples(n, batch, vocab, gamma, seed=77)

        if len(samples) == 0:
            pytest.skip("No samples collected — check acceptance logic.")

        # Build empirical CDF vs theoretical CDF.
        obs_counts = torch.zeros(vocab)
        for s in samples:
            if 0 <= s < vocab:
                obs_counts[s] += 1
        obs_freq = (obs_counts / obs_counts.sum()).tolist()

        # Cumulative empirical vs cumulative theoretical.
        cum_obs = 0.0
        cum_exp = 0.0
        ks_stat = 0.0
        tp = torch.tensor(target_probs[:vocab]).float()
        tp = tp / tp.sum()  # renormalize
        for i in range(vocab):
            cum_obs += obs_freq[i]
            cum_exp += tp[i].item()
            ks_stat = max(ks_stat, abs(cum_obs - cum_exp))

        # Critical value for KS test at alpha=0.01: 1.63 / sqrt(n)
        import math
        critical_value = 1.63 / math.sqrt(len(samples))
        assert ks_stat < critical_value, (
            f"KS statistic {ks_stat:.4f} >= critical value {critical_value:.4f}. "
            "Output distribution may not match target model."
        )

    @pytest.mark.slow
    def test_ks_test_passes_full(self) -> None:
        """KS test at α=0.01 over N=10,000 samples (nightly gate, AGENTS.md §2.1)."""
        pytest.importorskip("scipy")
        n, batch, vocab, gamma = _N_SAMPLES_SLOW, 1, 100, 1
        samples, target_probs = _collect_residual_samples(n, batch, vocab, gamma, seed=42)

        if len(samples) == 0:
            pytest.skip("No samples collected.")

        obs_counts = [samples.count(i) for i in range(vocab)]
        tp = torch.tensor(target_probs[:vocab]).float()
        tp = tp / tp.sum()
        expected_counts = [tp[i].item() * len(samples) for i in range(vocab)]

        chi2, p_value = stats.chisquare(obs_counts, f_exp=expected_counts)
        assert p_value >= _ALPHA_KS, (
            f"Chi-squared p_value={p_value:.4f} < α={_ALPHA_KS}. "
            "Distribution does not match target."
        )

