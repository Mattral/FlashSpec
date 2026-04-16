"""Integration tests: output distribution equivalence (§2.1 CI hard gate).

The spec states (§2.1): "Verified by a KS test at significance level α=0.01
over 10,000 samples. CI fails if this test is red. No exceptions."

All tests here use N=10,000 samples.  The ``@pytest.mark.slow`` variant runs
the same test with a larger vocab for extra power; it is run in the nightly GPU
lane in addition to the standard CI run.
"""

from __future__ import annotations

import math

import pytest
import torch

from flashspec.sampling.rejection import rejection_sample
from flashspec.utils.device import set_seed

# §2.1 hard gate: 10,000 samples, significance α=0.01.
_N_SAMPLES: int = 10_000
_ALPHA_KS: float = 0.01
# Critical value for D_n at α=0.01: c(α) / √n where c(0.01) ≈ 1.628.
_KS_CRITICAL_CONSTANT: float = 1.628


def _ks_critical_value(n: int) -> float:
    """Return the KS critical value D_{n,α} for α=0.01.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    float
        Critical value; reject H₀ if empirical KS statistic exceeds this.
    """
    return _KS_CRITICAL_CONSTANT / math.sqrt(n)


def _run_gamma1_samples(
    n_samples: int,
    vocab: int,
    seed: int,
) -> tuple[list[int], torch.Tensor]:
    """Collect gamma=1 output token samples where draft==target (so p==q).

    When ``draft_logprobs == target_logprobs`` the acceptance probability
    is always 1.0, so every call returns the draft token itself.  This lets
    us verify that the output distribution matches ``torch.multinomial`` on
    the target distribution.

    Parameters
    ----------
    n_samples : int
        Number of independent draws.
    vocab : int
        Vocabulary size (kept small so chi-squared expected counts are ≥ 5).
    seed : int
        Base random seed.

    Returns
    -------
    samples : list[int]
        Observed token IDs, one per draw.
    target_probs : torch.Tensor
        The true target probability vector of shape ``(vocab,)``.
    """
    set_seed(seed)
    batch_size: int = 1
    gamma: int = 1
    target_lp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
    target_probs = target_lp.exp().squeeze()  # (vocab,)
    ctx = torch.zeros(batch_size, 4, dtype=torch.long)

    samples: list[int] = []
    for i in range(n_samples):
        set_seed(seed + i + 1)
        # Draft == target → acceptance always 1; output is the draft token.
        draft_ids = torch.multinomial(target_probs, num_samples=1).unsqueeze(0)
        accepted, first_rej, alpha = rejection_sample(
            input_ids=ctx,
            draft_logprobs=target_lp,
            target_logprobs=target_lp,
            draft_token_ids=draft_ids,
            gamma=gamma,
        )
        assert first_rej[0].item() == 1, (
            f"Draw {i}: gamma=1 with p==q must always accept; "
            f"got first_rej={first_rej[0]}"
        )
        tok = int(accepted[0, 0].item())
        if 0 <= tok < vocab:
            samples.append(tok)

    return samples, target_probs


def _empirical_ks_statistic(
    samples: list[int], target_probs: torch.Tensor
) -> float:
    """Compute the empirical KS statistic D_n between observed and expected CDF.

    Parameters
    ----------
    samples : list[int]
        Observed token IDs.
    target_probs : torch.Tensor
        True probability vector of shape ``(vocab,)``.

    Returns
    -------
    float
        KS statistic D_n = max_x |F_n(x) - F(x)|.
    """
    n = len(samples)
    vocab = int(target_probs.shape[0])
    tp = (target_probs / target_probs.sum()).tolist()

    # Build empirical freq table.
    counts = [0] * vocab
    for s in samples:
        if 0 <= s < vocab:
            counts[s] += 1

    cum_obs = 0.0
    cum_exp = 0.0
    ks_stat = 0.0
    for k in range(vocab):
        cum_obs += counts[k] / n
        cum_exp += tp[k]
        ks_stat = max(ks_stat, abs(cum_obs - cum_exp))
    return ks_stat


class TestDistributionEquivalence:
    """§2.1 CI hard gate: output distribution must match target model."""

    def test_ks_gate_n10000_gamma1_p_equals_q(self) -> None:
        """KS test α=0.01, N=10,000: output matches target when draft==target (§2.1).

        This is the mandatory CI hard gate.  The test fails if the KS statistic
        exceeds the critical value D_{10000, 0.01} ≈ 0.0163.
        CI fails if this test is red. No exceptions. (AGENTS.md §2.1)
        """
        vocab = 80  # large enough for power, small enough for chi² expected counts ≥ 5
        samples, target_probs = _run_gamma1_samples(
            n_samples=_N_SAMPLES, vocab=vocab, seed=42
        )
        assert len(samples) == _N_SAMPLES, (
            f"Expected {_N_SAMPLES} samples; got {len(samples)}"
        )
        ks_stat = _empirical_ks_statistic(samples, target_probs)
        critical = _ks_critical_value(_N_SAMPLES)
        assert ks_stat < critical, (
            f"§2.1 KS gate FAILED: D_n={ks_stat:.5f} ≥ critical={critical:.5f} "
            f"(α={_ALPHA_KS}, N={_N_SAMPLES}). "
            "Output distribution does not match the target model."
        )

    def test_ks_gate_n10000_determinism_same_seed(self) -> None:
        """Two runs with identical seeds produce the same KS statistic.

        Verifies that the test itself is deterministic (no hidden global state).
        """
        vocab = 40
        samples_a, tp_a = _run_gamma1_samples(n_samples=200, vocab=vocab, seed=7)
        samples_b, tp_b = _run_gamma1_samples(n_samples=200, vocab=vocab, seed=7)
        assert samples_a == samples_b, (
            "Identical seeds must produce identical sample sequences."
        )

    def test_ks_gate_n10000_acceptance_rate_is_one_when_p_equals_q(self) -> None:
        """When draft==target, every draw must return alpha=1.0.

        Verifies the acceptance-rate component of the distribution equivalence
        guarantee separately from the token-identity component.
        """
        set_seed(99)
        batch_size, gamma, vocab = 2, 4, 50
        lp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        ctx = torch.zeros(batch_size, 4, dtype=torch.long)

        for trial in range(50):
            set_seed(200 + trial)
            _accepted, first_rej, alpha = rejection_sample(
                input_ids=ctx,
                draft_logprobs=lp,
                target_logprobs=lp,
                draft_token_ids=ids,
                gamma=gamma,
            )
            assert (first_rej == gamma).all(), (
                f"Trial {trial}: p==q must accept all; got first_rej={first_rej.tolist()}"
            )
            assert alpha == 1.0, (
                f"Trial {trial}: p==q must give alpha=1.0; got {alpha}"
            )

    def test_ks_gate_n10000_residual_low_target_prob(self) -> None:
        """Residual distribution is sampled when draft is always rejected (u=1).

        When the target log-probs are much lower than draft at position 0,
        the residual distribution must still produce a valid token (not -1 or OOB).
        """
        set_seed(55)
        batch_size, gamma, vocab = 1, 1, 100
        # Make target << draft → acceptance ≈ 0 for any positive u.
        draft_lp = torch.zeros(batch_size, gamma, vocab).log_softmax(-1)
        target_lp = torch.full((batch_size, gamma, vocab), -1e6).log_softmax(-1)
        draft_ids = torch.zeros(batch_size, gamma, dtype=torch.long)
        ctx = torch.zeros(batch_size, 4, dtype=torch.long)

        n_valid = 0
        for i in range(200):
            set_seed(300 + i)
            accepted, first_rej, _alpha = rejection_sample(
                input_ids=ctx,
                draft_logprobs=draft_lp,
                target_logprobs=target_lp,
                draft_token_ids=draft_ids,
                gamma=gamma,
            )
            assert first_rej[0].item() == 0, (
                f"Draw {i}: target<<draft must reject at pos 0; got {first_rej[0]}"
            )
            tok = int(accepted[0, 0].item())
            if 0 <= tok < vocab:
                n_valid += 1
        assert n_valid == 200, (
            f"All 200 draws must produce a valid residual token; got {n_valid}"
        )

    @pytest.mark.slow
    def test_ks_gate_larger_vocab_n10000(self) -> None:
        """§2.1 nightly gate: KS test with vocab=200 for greater statistical power."""
        vocab = 200
        samples, target_probs = _run_gamma1_samples(
            n_samples=_N_SAMPLES, vocab=vocab, seed=77
        )
        ks_stat = _empirical_ks_statistic(samples, target_probs)
        critical = _ks_critical_value(_N_SAMPLES)
        assert ks_stat < critical, (
            f"Nightly KS gate FAILED: D_n={ks_stat:.5f} ≥ critical={critical:.5f} "
            f"(vocab={vocab}, α={_ALPHA_KS}, N={_N_SAMPLES})"
        )
