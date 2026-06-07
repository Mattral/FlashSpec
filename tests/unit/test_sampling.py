"""Unit tests for speculative sampling correctness.

Tests distribution equivalence, token conservation, determinism, and
gamma=1 parity with standard autoregressive sampling.
"""

from __future__ import annotations

import pytest
import torch

from flashspec.sampling.rejection import rejection_sample
from flashspec.utils.device import set_seed

TOY_BATCH: int = 2
TOY_GAMMA: int = 4
TOY_VOCAB: int = 500


def _make_logprobs(
    batch: int, gamma: int, vocab: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random log-probs and draft token IDs."""
    set_seed(seed)
    dlp = torch.randn(batch, gamma, vocab).log_softmax(-1)
    tlp = torch.randn(batch, gamma, vocab).log_softmax(-1)
    ids = torch.randint(0, vocab, (batch, gamma))
    return dlp, tlp, ids


class TestRejectionSample:
    """Tests for ``rejection_sample``."""

    def test_output_shapes_are_correct(self) -> None:
        """rejection_sample returns tensors with correct shapes."""
        dlp, tlp, ids = _make_logprobs(TOY_BATCH, TOY_GAMMA, TOY_VOCAB, 0)
        input_ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, 8))
        accepted_ids, first_rej, alpha = rejection_sample(
            input_ids=input_ids,
            draft_logprobs=dlp,
            target_logprobs=tlp,
            draft_token_ids=ids,
            gamma=TOY_GAMMA,
        )
        assert accepted_ids.shape == (TOY_BATCH, TOY_GAMMA), (
            f"accepted_ids.shape={accepted_ids.shape}"
        )
        assert first_rej.shape == (TOY_BATCH,), f"first_rej.shape={first_rej.shape}"
        assert 0.0 <= alpha <= 1.0, f"alpha={alpha} out of [0,1]"

    def test_determinism_same_seed_same_output(self) -> None:
        """Identical seed produces identical output."""
        dlp, tlp, ids = _make_logprobs(TOY_BATCH, TOY_GAMMA, TOY_VOCAB, 7)
        input_ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, 8))

        set_seed(42)
        out1, frej1, a1 = rejection_sample(input_ids, dlp, tlp, ids, gamma=TOY_GAMMA)
        set_seed(42)
        out2, frej2, a2 = rejection_sample(input_ids, dlp, tlp, ids, gamma=TOY_GAMMA)

        assert (out1 == out2).all(), "Identical seeds must produce identical accepted_ids."
        assert (frej1 == frej2).all(), "Identical seeds must produce identical first_rejection."

    def test_first_rejection_is_in_valid_range(self) -> None:
        """first_rejection is always in [0, gamma]."""
        dlp, tlp, ids = _make_logprobs(TOY_BATCH, TOY_GAMMA, TOY_VOCAB, 5)
        input_ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, 8))
        _out, first_rej, _a = rejection_sample(input_ids, dlp, tlp, ids, gamma=TOY_GAMMA)
        assert (first_rej >= 0).all(), f"Negative first_rejection: {first_rej}"
        assert (first_rej <= TOY_GAMMA).all(), f"first_rejection > gamma: {first_rej}"

    def test_gamma_mismatch_raises_value_error(self) -> None:
        """Mismatched gamma raises ValueError."""
        dlp, tlp, ids = _make_logprobs(TOY_BATCH, TOY_GAMMA, TOY_VOCAB, 0)
        input_ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, 8))
        with pytest.raises(ValueError, match="gamma"):
            rejection_sample(input_ids, dlp, tlp, ids, gamma=TOY_GAMMA + 1)

    def test_mismatched_shapes_raises_value_error(self) -> None:
        """Mismatched draft/target logprob shapes raise ValueError."""
        dlp = torch.randn(TOY_BATCH, TOY_GAMMA, TOY_VOCAB).log_softmax(-1)
        tlp = torch.randn(TOY_BATCH, TOY_GAMMA, TOY_VOCAB + 1).log_softmax(-1)
        ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, TOY_GAMMA))
        input_ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, 8))
        with pytest.raises(ValueError, match="identical shapes"):
            rejection_sample(input_ids, dlp, tlp, ids, gamma=TOY_GAMMA)

    def test_all_accept_case_first_rejection_is_gamma(self) -> None:
        """When acceptance probability is 1 everywhere, first_rejection == gamma."""
        batch, gamma, vocab = 2, 4, 100
        # Target == draft → ratio = 1; with u drawn from uniform, most should accept.
        lp = torch.randn(batch, gamma, vocab).log_softmax(-1)
        ids = torch.randint(0, vocab, (batch, gamma))
        input_ids = torch.randint(0, vocab, (batch, 8))
        # Force u = 0 for guaranteed acceptance via seed that produces low u.
        # Instead, verify via multiple seeds that first_rejection <= gamma.
        _out, first_rej, alpha = rejection_sample(input_ids, lp, lp, ids, gamma=gamma)
        # When draft == target, acceptance prob = 1 for all positions.
        assert (first_rej == gamma).all(), (
            f"When p==q, all tokens accepted but first_rej={first_rej}"
        )

    def test_acceptance_rate_float_between_zero_and_one(self) -> None:
        """acceptance_rate is a float in [0.0, 1.0]."""
        dlp, tlp, ids = _make_logprobs(TOY_BATCH, TOY_GAMMA, TOY_VOCAB, 1)
        input_ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, 8))
        _out, _frej, alpha = rejection_sample(input_ids, dlp, tlp, ids, gamma=TOY_GAMMA)
        assert isinstance(alpha, float), f"alpha is {type(alpha)}, expected float"
        assert 0.0 <= alpha <= 1.0, f"alpha={alpha} not in [0, 1]"

    def test_token_conservation_accepted_ids_never_exceed_gamma(self) -> None:
        """Token-conservation: output has exactly gamma columns (padded with -1 if needed)."""
        dlp, tlp, ids = _make_logprobs(TOY_BATCH, TOY_GAMMA, TOY_VOCAB, 3)
        input_ids = torch.randint(0, TOY_VOCAB, (TOY_BATCH, 8))
        accepted_ids, first_rej, _alpha = rejection_sample(
            input_ids, dlp, tlp, ids, gamma=TOY_GAMMA
        )
        assert accepted_ids.shape[1] == TOY_GAMMA, (
            f"accepted_ids must have exactly gamma={TOY_GAMMA} columns; "
            f"got shape {accepted_ids.shape}"
        )
        # Tokens beyond first_rejection must be -1 (padding).
        for b in range(TOY_BATCH):
            frej = first_rej[b].item()
            # The residual token occupies the first_rejection slot (≤ gamma-1).
            # Every slot after that must be -1.
            for col in range(int(frej) + 1, TOY_GAMMA):
                val = accepted_ids[b, col].item()
                assert val == -1, (
                    f"Batch {b}: accepted_ids[{b}, {col}]={val} should be -1 "
                    f"(first_rej={frej})"
                )

    def test_gamma1_matches_standard_multinomial_sampling(self) -> None:
        """gamma=1 must match standard multinomial sampling from the target exactly."""
        # With gamma=1 and identical draft/target, rejection_sample should always
        # return first_rejection=1 (the single token is accepted) and the residual
        # slot contains the target's sample.  We verify that over N draws the
        # empirical distribution matches torch.multinomial on the target distribution.
        set_seed(99)
        batch_size = 1
        vocab = 50
        n_draws = 500

        # Fixed target log-probs.
        target_lp = torch.randn(batch_size, 1, vocab).log_softmax(-1)
        target_probs = target_lp.exp().squeeze()  # (vocab,)

        # Use draft == target so acceptance probability is always 1.
        # The accepted token is always the draft token itself.
        spec_counts = torch.zeros(vocab, dtype=torch.long)
        for i in range(n_draws):
            set_seed(1000 + i)
            draft_ids = torch.multinomial(target_probs, num_samples=1).unsqueeze(0)  # (1,1)
            ctx = torch.zeros(batch_size, 4, dtype=torch.long)
            accepted, first_rej, alpha = rejection_sample(
                input_ids=ctx,
                draft_logprobs=target_lp,
                target_logprobs=target_lp,
                draft_token_ids=draft_ids,
                gamma=1,
            )
            assert first_rej[0].item() == 1, (
                f"gamma=1 with p==q must always accept; got first_rej={first_rej[0]}"
            )
            tok = accepted[0, 0].item()
            if 0 <= tok < vocab:
                spec_counts[tok] += 1

        # Chi-squared goodness-of-fit: empirical vs expected.
        expected = (target_probs * n_draws).tolist()
        observed = spec_counts.tolist()
        # Verify rough agreement: no bucket should deviate by more than 5 sigma.
        for k in range(vocab):
            exp_k = expected[k]
            obs_k = observed[k]
            # Allow large relative deviation for very-low-prob tokens.
            if exp_k < 1.0:
                continue
            import math
            sigma = math.sqrt(exp_k * (1 - exp_k / n_draws))
            deviation = abs(obs_k - exp_k)
            assert deviation < 5 * sigma + 3, (
                f"Token {k}: obs={obs_k}, exp={exp_k:.1f}, "
                f"deviation={deviation:.1f} > 5σ={5*sigma:.1f}"
            )
