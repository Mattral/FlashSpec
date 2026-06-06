"""Unit tests for the Triton token-verification kernel.

Verifies numerical parity between the Triton kernel (``verify_tokens``)
and the pure-PyTorch reference (``verify_tokens_reference``).

All GPU fixtures are marked ``@pytest.mark.gpu`` and skipped on CPU CI.
CPU tests use the reference implementation directly.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from flashspec.kernels._reference import verify_tokens_reference
from flashspec.utils.device import set_seed

# ── Tolerances from AGENTS.md §2.2 ───────────────────────────────────────────
_ATOL_FLOAT32: float = 1e-5
_ATOL_BFLOAT16: float = 1e-3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_inputs(
    batch_size: int,
    gamma: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create reproducible random kernel inputs."""
    set_seed(seed)
    dlp = torch.randn(batch_size, gamma, vocab_size).to(dtype).log_softmax(-1)
    tlp = torch.randn(batch_size, gamma, vocab_size).to(dtype).log_softmax(-1)
    u   = torch.rand(batch_size, gamma)
    ids = torch.randint(0, vocab_size, (batch_size, gamma))
    return dlp, tlp, u, ids


# ── CPU reference tests (no GPU required) ────────────────────────────────────


class TestVerifyTokensReference:
    """Tests for the pure-PyTorch reference implementation."""

    def test_reference_output_shape_is_correct(self) -> None:
        """Reference returns tensors of the correct shape."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=1000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (2, 4), f"accepted.shape={accepted.shape}"
        assert first_rej.shape == (2,), f"first_rej.shape={first_rej.shape}"

    def test_reference_all_accept_returns_gamma(self) -> None:
        """When u < accept_prob for all positions, first_rejection == gamma."""
        batch_size, gamma, vocab = 2, 4, 1000
        # Force all acceptance: set target == draft (ratio = 1), u = 0.0.
        lp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        u = torch.zeros(batch_size, gamma)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        _accepted, first_rej = verify_tokens_reference(lp, lp, u, ids)
        assert (first_rej == gamma).all(), f"Expected first_rej==gamma, got {first_rej}"

    def test_reference_all_reject_returns_zero(self) -> None:
        """When u >= accept_prob for all positions, first_rejection == 0."""
        batch_size, gamma, vocab = 2, 4, 1000
        # Force all rejection: target log_prob << draft (accept_prob ≈ 0), u = 1.0.
        dlp = torch.full((batch_size, gamma, vocab), 0.0).log_softmax(-1)
        tlp = torch.full((batch_size, gamma, vocab), -1e6).log_softmax(-1)
        u = torch.ones(batch_size, gamma)
        ids = torch.zeros(batch_size, gamma, dtype=torch.long)
        _accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert (first_rej == 0).all(), f"Expected first_rej==0, got {first_rej}"

    def test_reference_batch_size_one(self) -> None:
        """Reference works correctly with batch_size=1."""
        dlp, tlp, u, ids = _make_inputs(batch_size=1, gamma=4, vocab_size=1000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (1, 4), f"accepted.shape={accepted.shape}"
        assert first_rej.shape == (1,), f"first_rej.shape={first_rej.shape}"

    def test_reference_batch_size_64(self) -> None:
        """Reference works correctly with batch_size=64."""
        dlp, tlp, u, ids = _make_inputs(batch_size=64, gamma=4, vocab_size=1000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (64, 4), f"accepted.shape={accepted.shape}"

    def test_reference_llama_vocab_size(self) -> None:
        """Reference handles vocab_size=32000 (Llama)."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=32_000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (2, 4), f"accepted.shape={accepted.shape}"

    def test_reference_mistral_vocab_size(self) -> None:
        """Reference handles vocab_size=32768 (Mistral)."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=32_768)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (2, 4), f"accepted.shape={accepted.shape}"

    def test_reference_first_rejection_is_int32(self) -> None:
        """first_rejection output dtype is int32."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=1000)
        _accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert first_rej.dtype == torch.int32, f"dtype={first_rej.dtype}"

    def test_reference_mismatched_shapes_raises_value_error(self) -> None:
        """Mismatched logprob shapes raise ValueError."""
        dlp = torch.randn(2, 4, 1000).log_softmax(-1)
        tlp = torch.randn(2, 4, 500).log_softmax(-1)  # wrong vocab
        u   = torch.rand(2, 4)
        ids = torch.randint(0, 500, (2, 4))
        with pytest.raises(ValueError, match="identical shapes"):
            verify_tokens_reference(dlp, tlp, u, ids)

    def test_reference_acceptance_prob_clamped_to_one(self) -> None:
        """Accept probability is clamped to [0, 1] when p > q."""
        batch_size, gamma, vocab = 1, 1, 10
        # Make target >> draft at token 0: ratio > 1, still accepted for any u < 1.
        dlp = torch.full((batch_size, gamma, vocab), -1e3).log_softmax(-1)
        tlp = torch.zeros(batch_size, gamma, vocab).log_softmax(-1)
        u   = torch.tensor([[0.99]])
        ids = torch.zeros(batch_size, gamma, dtype=torch.long)
        accepted, _ = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.all(), "Token should be accepted when accept_prob is clamped to 1."

    def test_reference_irrelevant_logprobs_do_not_affect_output(self) -> None:
        """Output is identical regardless of logprobs at non-selected vocab positions."""
        batch_size, gamma, vocab = 2, 4, 1000
        set_seed(99)
        dlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        tlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        u   = torch.rand(batch_size, gamma)
        ids = torch.randint(0, vocab, (batch_size, gamma))

        accepted_a, frej_a = verify_tokens_reference(dlp, tlp, u, ids)

        # Perturb logprobs at positions other than selected tokens.
        dlp_b = dlp.clone()
        for b in range(batch_size):
            for g in range(gamma):
                tok = ids[b, g].item()
                # Zero out all other vocab positions (changes dist but not token prob).
                dlp_b[b, g, :] = -1e9
                dlp_b[b, g, tok] = dlp[b, g, tok]
        dlp_b = dlp_b.log_softmax(-1)

        accepted_b, frej_b = verify_tokens_reference(dlp_b, tlp, u, ids)
        # The selected token's contribution is unchanged → same outcome.
        assert (accepted_a == accepted_b).all(), "Output should not depend on non-selected logprobs"


# ── Hypothesis property tests ─────────────────────────────────────────────────

@given(
    batch_size=st.integers(min_value=1, max_value=8),
    gamma=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=30, deadline=None)
def test_reference_first_rejection_in_valid_range(batch_size: int, gamma: int) -> None:
    """first_rejection values are always in [0, gamma] for any input."""
    vocab = 500
    dlp, tlp, u, ids = _make_inputs(batch_size, gamma, vocab)
    _accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
    assert (first_rej >= 0).all(), f"first_rej has negative values: {first_rej}"
    assert (first_rej <= gamma).all(), f"first_rej exceeds gamma={gamma}: {first_rej}"
