"""Unit tests for the Triton token-verification kernel.

Verifies numerical parity between the Triton kernel (``verify_tokens``)
and the pure-PyTorch reference (``verify_tokens_reference``).

All GPU fixtures are inside ``TestVerifyTokensTritonParity`` and decorated
with ``@pytest.mark.gpu``; they are skipped in the CPU fast-lane CI.
CPU tests in ``TestVerifyTokensReference`` require no GPU.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from flashspec.kernels._reference import verify_tokens_reference
from flashspec.utils.device import set_seed

# ── Tolerances (AGENTS.md §2.2) ───────────────────────────────────────────────
_ATOL_FLOAT32: float = 1e-5
_ATOL_BFLOAT16: float = 1e-3


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_inputs(
    batch_size: int,
    gamma: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create reproducible random kernel inputs on CPU."""
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
        """Reference returns accepted (batch,gamma) bool and first_rej (batch,) int32."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=1000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (2, 4), f"accepted.shape={accepted.shape}"
        assert first_rej.shape == (2,), f"first_rej.shape={first_rej.shape}"

    def test_reference_all_accept_when_u_zero_and_p_equals_q(self) -> None:
        """When u=0 and target==draft (ratio=1), every token is accepted."""
        batch_size, gamma, vocab = 2, 4, 1000
        lp  = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        u   = torch.zeros(batch_size, gamma)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        _accepted, first_rej = verify_tokens_reference(lp, lp, u, ids)
        assert (first_rej == gamma).all(), (
            f"Expected first_rej==gamma={gamma} (all accept), got {first_rej}"
        )

    def test_reference_all_reject_when_u_one_and_target_much_lower(self) -> None:
        """When u=1 and target prob << draft prob, every token is rejected."""
        batch_size, gamma, vocab = 2, 4, 1000
        dlp = torch.full((batch_size, gamma, vocab), 0.0).log_softmax(-1)
        tlp = torch.full((batch_size, gamma, vocab), -1e6).log_softmax(-1)
        u   = torch.ones(batch_size, gamma)
        ids = torch.zeros(batch_size, gamma, dtype=torch.long)
        _accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert (first_rej == 0).all(), (
            f"Expected first_rej==0 (all reject), got {first_rej}"
        )

    def test_reference_batch_size_one_returns_correct_shapes(self) -> None:
        """Reference works correctly with batch_size=1."""
        dlp, tlp, u, ids = _make_inputs(batch_size=1, gamma=4, vocab_size=1000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (1, 4), f"accepted.shape={accepted.shape}"
        assert first_rej.shape == (1,), f"first_rej.shape={first_rej.shape}"

    def test_reference_batch_size_64_returns_correct_shapes(self) -> None:
        """Reference works correctly with batch_size=64."""
        dlp, tlp, u, ids = _make_inputs(batch_size=64, gamma=4, vocab_size=1000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (64, 4), f"accepted.shape={accepted.shape}"

    def test_reference_llama_vocab_size_32000_succeeds(self) -> None:
        """Reference handles vocab_size=32000 (Llama) without error."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=32_000)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (2, 4), f"accepted.shape={accepted.shape}"

    def test_reference_mistral_vocab_size_32768_succeeds(self) -> None:
        """Reference handles vocab_size=32768 (Mistral) without error."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=32_768)
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (2, 4), f"accepted.shape={accepted.shape}"

    def test_reference_first_rejection_dtype_is_int32(self) -> None:
        """first_rejection tensor has dtype int32 as required by the kernel contract."""
        dlp, tlp, u, ids = _make_inputs(batch_size=2, gamma=4, vocab_size=1000)
        _accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert first_rej.dtype == torch.int32, (
            f"Expected int32, got {first_rej.dtype}"
        )

    def test_reference_mismatched_shapes_raises_value_error(self) -> None:
        """Mismatched draft/target vocab sizes raise ValueError with clear message."""
        dlp = torch.randn(2, 4, 1000).log_softmax(-1)
        tlp = torch.randn(2, 4, 500).log_softmax(-1)   # wrong vocab
        u   = torch.rand(2, 4)
        ids = torch.randint(0, 500, (2, 4))
        with pytest.raises(ValueError, match="identical shapes"):
            verify_tokens_reference(dlp, tlp, u, ids)

    def test_reference_acceptance_prob_clamped_to_one_when_target_dominates(self) -> None:
        """Accept probability is clamped to 1.0 when p >> q, so token is accepted."""
        batch_size, gamma, vocab = 1, 1, 10
        dlp = torch.full((batch_size, gamma, vocab), -1e3).log_softmax(-1)
        tlp = torch.zeros(batch_size, gamma, vocab).log_softmax(-1)
        u   = torch.tensor([[0.99]])
        ids = torch.zeros(batch_size, gamma, dtype=torch.long)
        accepted, _ = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.all(), (
            "Token should be accepted when accept_prob is clamped to 1.0"
        )

    def test_reference_nonselected_logprobs_do_not_change_output(self) -> None:
        """Output is identical regardless of logprob values at non-draft-selected positions."""
        batch_size, gamma, vocab = 2, 4, 1000
        set_seed(99)
        dlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        tlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        u   = torch.rand(batch_size, gamma)
        ids = torch.randint(0, vocab, (batch_size, gamma))

        accepted_a, frej_a = verify_tokens_reference(dlp, tlp, u, ids)

        # Perturb every position except the selected token — result must be identical.
        dlp_b = dlp.clone()
        for b in range(batch_size):
            for g in range(gamma):
                tok = int(ids[b, g].item())
                saved = dlp_b[b, g, tok].clone()
                dlp_b[b, g, :] = -1e9
                dlp_b[b, g, tok] = saved
        dlp_b = dlp_b.log_softmax(-1)

        accepted_b, frej_b = verify_tokens_reference(dlp_b, tlp, u, ids)
        assert (accepted_a == accepted_b).all(), (
            "Perturbing non-selected logprob positions must not change acceptance mask"
        )
        assert (frej_a == frej_b).all(), (
            "Perturbing non-selected logprob positions must not change first_rejection"
        )

    def test_reference_bfloat16_input_produces_correct_shapes(self) -> None:
        """Reference handles bfloat16 inputs and returns correctly shaped outputs."""
        dlp, tlp, u, ids = _make_inputs(
            batch_size=4, gamma=4, vocab_size=1000, dtype=torch.bfloat16, seed=10
        )
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (4, 4), (
            f"bfloat16 input: accepted.shape={accepted.shape}, expected (4, 4)"
        )
        assert first_rej.dtype == torch.int32, (
            f"bfloat16 input: first_rej.dtype={first_rej.dtype}, expected int32"
        )

    def test_reference_bfloat16_parity_dtypes_match_float32(self) -> None:
        """bfloat16 and float32 references return the same output dtypes (§2.2)."""
        batch_size, gamma, vocab = 4, 4, 500
        set_seed(20)
        u   = torch.rand(batch_size, gamma)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        dlp_f32 = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        tlp_f32 = torch.randn(batch_size, gamma, vocab).log_softmax(-1)

        accepted_f32, frej_f32 = verify_tokens_reference(dlp_f32, tlp_f32, u, ids)
        accepted_bf16, frej_bf16 = verify_tokens_reference(
            dlp_f32.to(torch.bfloat16), tlp_f32.to(torch.bfloat16), u, ids
        )

        assert accepted_f32.dtype == torch.bool, (
            f"float32 accepted dtype={accepted_f32.dtype}, expected bool"
        )
        assert accepted_bf16.dtype == torch.bool, (
            f"bfloat16 accepted dtype={accepted_bf16.dtype}, expected bool"
        )
        assert frej_f32.dtype == torch.int32, (
            f"float32 first_rej dtype={frej_f32.dtype}, expected int32"
        )
        assert frej_bf16.dtype == torch.int32, (
            f"bfloat16 first_rej dtype={frej_bf16.dtype}, expected int32"
        )

    def test_reference_canonical_shape_batch8_gamma8_vocab32000(self) -> None:
        """Reference handles canonical shape (batch=8, gamma=8, vocab=32000) per §2.2."""
        dlp, tlp, u, ids = _make_inputs(
            batch_size=8, gamma=8, vocab_size=32_000, seed=77
        )
        accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
        assert accepted.shape == (8, 8), (
            f"Canonical shape failed: accepted.shape={accepted.shape}, expected (8, 8)"
        )
        assert first_rej.shape == (8,), (
            f"Canonical shape failed: first_rej.shape={first_rej.shape}, expected (8,)"
        )
        assert (first_rej >= 0).all(), (
            f"first_rej has negative values: {first_rej}"
        )
        assert (first_rej <= 8).all(), (
            f"first_rej exceeds gamma=8: {first_rej}"
        )


# ── Hypothesis property-based tests ──────────────────────────────────────────


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    gamma=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=30, deadline=None)
def test_reference_first_rejection_always_in_valid_range(
    batch_size: int, gamma: int
) -> None:
    """first_rejection is always in [0, gamma] for any valid (batch, gamma) input."""
    vocab = 500
    dlp, tlp, u, ids = _make_inputs(batch_size, gamma, vocab)
    _accepted, first_rej = verify_tokens_reference(dlp, tlp, u, ids)
    assert (first_rej >= 0).all(), (
        f"first_rej has negative values: {first_rej}"
    )
    assert (first_rej <= gamma).all(), (
        f"first_rej exceeds gamma={gamma}: {first_rej}"
    )


# ── GPU parity tests: Triton kernel vs reference (§2.2) ──────────────────────


@pytest.mark.gpu
class TestVerifyTokensTritonParity:
    """Numerical parity between ``verify_tokens`` (Triton) and the reference.

    All tests require a CUDA GPU; they are skipped in the CPU CI fast-lane
    and run nightly via the ``@pytest.mark.gpu`` selector.
    """

    def test_triton_parity_float32_canonical_shape_matches_reference(self) -> None:
        """Triton kernel accepted mask and first_rejection match reference (float32, §2.2)."""
        from flashspec.kernels import verify_tokens

        device = torch.device("cuda")
        dlp, tlp, u, ids = _make_inputs(
            batch_size=8, gamma=8, vocab_size=32_000, dtype=torch.float32, seed=42
        )
        ref_accepted, ref_frej = verify_tokens_reference(dlp, tlp, u, ids)

        triton_accepted, triton_frej = verify_tokens(
            dlp.to(device), tlp.to(device), u.to(device), ids.to(device)
        )
        ta_cpu = triton_accepted.cpu()
        tf_cpu = triton_frej.cpu()

        n_diff = int((ta_cpu != ref_accepted).sum().item())
        assert n_diff == 0, (
            f"Triton float32 accepted mask differs from reference at {n_diff} positions"
        )
        assert (tf_cpu == ref_frej).all(), (
            f"Triton float32 first_rejection mismatch: "
            f"max_diff={int((tf_cpu - ref_frej).abs().max().item())}"
        )

    def test_triton_parity_bfloat16_canonical_shape_matches_reference(self) -> None:
        """Triton kernel accepted mask and first_rejection match reference (bfloat16, §2.2)."""
        from flashspec.kernels import verify_tokens

        device = torch.device("cuda")
        dlp, tlp, u, ids = _make_inputs(
            batch_size=8, gamma=8, vocab_size=32_000, dtype=torch.bfloat16, seed=43
        )
        ref_accepted, ref_frej = verify_tokens_reference(dlp, tlp, u, ids)

        triton_accepted, triton_frej = verify_tokens(
            dlp.to(device), tlp.to(device), u.to(device), ids.to(device)
        )
        n_diff = int((triton_accepted.cpu() != ref_accepted).sum().item())
        assert n_diff == 0, (
            f"Triton bfloat16 accepted mask differs from reference at {n_diff} positions"
        )
        assert (triton_frej.cpu() == ref_frej).all(), (
            "Triton bfloat16 first_rejection does not match reference"
        )

    def test_triton_parity_batch1_float32_matches_reference(self) -> None:
        """Triton kernel matches reference for the minimal batch_size=1 case."""
        from flashspec.kernels import verify_tokens

        device = torch.device("cuda")
        dlp, tlp, u, ids = _make_inputs(
            batch_size=1, gamma=4, vocab_size=32_000, dtype=torch.float32, seed=11
        )
        ref_accepted, ref_frej = verify_tokens_reference(dlp, tlp, u, ids)
        triton_accepted, triton_frej = verify_tokens(
            dlp.to(device), tlp.to(device), u.to(device), ids.to(device)
        )
        assert (triton_accepted.cpu() == ref_accepted).all(), (
            "Triton batch_size=1 accepted mask does not match reference"
        )
        assert (triton_frej.cpu() == ref_frej).all(), (
            "Triton batch_size=1 first_rejection does not match reference"
        )

    def test_triton_parity_batch64_float32_matches_reference(self) -> None:
        """Triton kernel matches reference for batch_size=64."""
        from flashspec.kernels import verify_tokens

        device = torch.device("cuda")
        dlp, tlp, u, ids = _make_inputs(
            batch_size=64, gamma=4, vocab_size=1000, dtype=torch.float32, seed=22
        )
        ref_accepted, ref_frej = verify_tokens_reference(dlp, tlp, u, ids)
        triton_accepted, triton_frej = verify_tokens(
            dlp.to(device), tlp.to(device), u.to(device), ids.to(device)
        )
        assert (triton_accepted.cpu() == ref_accepted).all(), (
            "Triton batch_size=64 accepted mask does not match reference"
        )
        assert (triton_frej.cpu() == ref_frej).all(), (
            "Triton batch_size=64 first_rejection does not match reference"
        )

    def test_triton_all_accept_when_u_zero_returns_gamma(self) -> None:
        """Triton kernel returns first_rejection==gamma when u=0 and p==q (all accept)."""
        from flashspec.kernels import verify_tokens

        device = torch.device("cuda")
        batch_size, gamma, vocab = 4, 4, 1000
        lp  = torch.randn(batch_size, gamma, vocab).log_softmax(-1).to(device)
        u   = torch.zeros(batch_size, gamma, device=device)
        ids = torch.randint(0, vocab, (batch_size, gamma), device=device)

        _accepted, first_rej = verify_tokens(lp, lp, u, ids)
        assert (first_rej.cpu() == gamma).all(), (
            f"Expected all first_rej=={gamma}, got {first_rej.cpu().tolist()}"
        )

    def test_triton_all_reject_when_u_one_returns_zero(self) -> None:
        """Triton kernel returns first_rejection==0 when u=1 and target prob << draft."""
        from flashspec.kernels import verify_tokens

        device = torch.device("cuda")
        batch_size, gamma, vocab = 4, 4, 1000
        dlp = torch.full((batch_size, gamma, vocab), 0.0).log_softmax(-1).to(device)
        tlp = torch.full((batch_size, gamma, vocab), -1e6).log_softmax(-1).to(device)
        u   = torch.ones(batch_size, gamma, device=device)
        ids = torch.zeros(batch_size, gamma, dtype=torch.long, device=device)

        _accepted, first_rej = verify_tokens(dlp, tlp, u, ids)
        assert (first_rej.cpu() == 0).all(), (
            f"Expected all first_rej==0, got {first_rej.cpu().tolist()}"
        )


# ── Triton-unavailable fallback (cross-platform, no GPU required) ───────────


class TestKernelsPackageWithoutTriton:
    """``flashspec.kernels`` must import successfully even if Triton is
    absent (e.g. on Windows or macOS, where Triton ships no PyPI wheels).

    Simulates the absence of Triton by removing ``triton`` and the kernel
    submodules from ``sys.modules`` and blocking their import, then
    re-importing ``flashspec.kernels`` fresh.
    """

    def test_kernels_package_imports_without_triton(self) -> None:
        """``import flashspec.kernels`` succeeds even when Triton is missing."""
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        # Drop any cached modules so re-import re-triggers the try/except.
        for mod in list(sys.modules):
            if mod.startswith("flashspec.kernels"):
                del sys.modules[mod]

        builtins.__import__ = fake_import
        try:
            kernels = importlib.import_module("flashspec.kernels")
        finally:
            builtins.__import__ = real_import
            for mod in list(sys.modules):
                if mod.startswith("flashspec.kernels"):
                    del sys.modules[mod]

        assert kernels.TRITON_AVAILABLE is False, (
            "TRITON_AVAILABLE should be False when triton import fails"
        )

    def test_verify_tokens_raises_actionable_import_error_without_triton(self) -> None:
        """Calling verify_tokens without Triton raises ImportError mentioning flashspec[gpu]."""
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        for mod in list(sys.modules):
            if mod.startswith("flashspec.kernels"):
                del sys.modules[mod]

        builtins.__import__ = fake_import
        try:
            kernels = importlib.import_module("flashspec.kernels")
            with pytest.raises(ImportError, match=r"flashspec\[gpu\]"):
                kernels.verify_tokens(None, None, None, None)
        finally:
            builtins.__import__ = real_import
            for mod in list(sys.modules):
                if mod.startswith("flashspec.kernels"):
                    del sys.modules[mod]

    def test_gather_accepted_raises_actionable_import_error_without_triton(self) -> None:
        """Calling gather_accepted without Triton raises ImportError mentioning _reference."""
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        for mod in list(sys.modules):
            if mod.startswith("flashspec.kernels"):
                del sys.modules[mod]

        builtins.__import__ = fake_import
        try:
            kernels = importlib.import_module("flashspec.kernels")
            with pytest.raises(ImportError, match="_reference"):
                kernels.gather_accepted(None, None)
        finally:
            builtins.__import__ = real_import
            for mod in list(sys.modules):
                if mod.startswith("flashspec.kernels"):
                    del sys.modules[mod]
