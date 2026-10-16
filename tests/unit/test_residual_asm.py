"""Residual backend selection and the documented NaN divergence.

The AVX2 kernel is a Linux ELF library. These tests exercise the
selection logic with a fake module and check that finite inputs routed
through a kernel which implements the same formula match PyTorch.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

import flashspec.sampling.rejection as rejection
from flashspec.sampling.rejection import (
    ASM_NAN_CLAMP_TO_ZERO,
    _asm_row_ready,
    _resolve_asm_backend,
    active_residual_backend,
    residual_distribution,
)
from flashspec.utils.cpu import cpu_has_avx2
from flashspec.utils.device import set_seed


def _finite_pair(batch: int = 2, vocab: int = 17) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two finite probability rows."""
    generator = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(batch, vocab, generator=generator), dim=-1)
    q = torch.softmax(torch.randn(batch, vocab, generator=generator), dim=-1)
    return p, q


class TestPytorchResidual:
    """The reference formula stays available on every platform."""

    def test_matches_clamped_normalisation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Finite inputs match max(0, p-q) / sum with the 1e-9 floor."""
        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", False)
        p, q = _finite_pair()
        got = residual_distribution(p, q)
        ref = torch.clamp(p - q, min=0.0)
        ref = ref / ref.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        assert torch.allclose(got, ref, atol=1e-6)

    def test_nan_is_preserved_by_pytorch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """torch.clamp keeps NaN. The assembly kernel does not; see the flag."""
        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", False)
        p = torch.tensor([float("nan"), 0.2, 0.8])
        q = torch.tensor([0.0, 0.1, 0.1])
        got = residual_distribution(p, q)
        assert torch.isnan(got[0])
        assert ASM_NAN_CLAMP_TO_ZERO is True

    def test_shape_mismatch_raises(self) -> None:
        """Different shapes are rejected before any kernel call."""
        with pytest.raises(ValueError, match="same shape"):
            residual_distribution(torch.zeros(3), torch.zeros(4))

    def test_noncontiguous_input_skips_asm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-contiguous CPU tensor stays on the PyTorch path."""
        calls = {"n": 0}

        def _count(p: object, q: object, out: torch.Tensor) -> None:
            calls["n"] += 1
            out[...] = 0

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _count)
        base = torch.randn(4, 8)
        p = base.t()
        q = torch.rand_like(base).t()
        assert not p.is_contiguous()
        got = residual_distribution(p, q)
        assert calls["n"] == 0
        assert got.shape == p.shape

    def test_kernel_exception_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A kernel that raises does not escape to the caller."""

        def _boom(p: object, q: object, out: object) -> None:
            raise RuntimeError("kernel failed")

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _boom)
        p, q = _finite_pair(batch=1, vocab=8)
        got = residual_distribution(p, q)
        assert torch.isfinite(got).all()

    def test_matching_kernel_agrees_with_pytorch_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kernel that implements the same formula yields the same tokens."""
        import numpy as np

        def _same(p: object, q: object, out: torch.Tensor) -> None:
            pt = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))
            qt = torch.from_numpy(np.ascontiguousarray(q, dtype=np.float32))
            ref = torch.clamp(pt - qt, min=0.0)
            ref = ref / ref.sum().clamp(min=1e-9)
            out[...] = ref.numpy()

        batch, gamma, vocab = 1, 4, 32
        draft = torch.randn(batch, gamma, vocab).log_softmax(-1)
        target = torch.randn(batch, gamma, vocab).log_softmax(-1)
        token_ids = torch.randint(0, vocab, (batch, gamma))
        context = torch.randint(0, vocab, (batch, 4))

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", False)
        set_seed(11)
        ids_py, rej_py, _alpha_py = rejection.rejection_sample(
            context, draft, target, token_ids, gamma=gamma
        )
        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _same)
        set_seed(11)
        ids_asm, rej_asm, _alpha_asm = rejection.rejection_sample(
            context, draft, target, token_ids, gamma=gamma
        )
        assert torch.equal(ids_py, ids_asm)
        assert torch.equal(rej_py, rej_asm)


class TestBackendResolution:
    """Import-time selection never runs a mismatched binary."""

    def test_missing_package_is_pytorch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing module selects PyTorch."""
        monkeypatch.setitem(sys.modules, "flashspec_asm_kernel", None)
        available, fn, reason = _resolve_asm_backend()
        assert available is False
        assert fn is None
        assert "not installed" in reason

    def test_abi_mismatch_warns_and_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ABI 99 is not executed."""
        module = types.ModuleType("flashspec_asm_kernel")
        module.abi_version = lambda: 99  # type: ignore[attr-defined]
        module.cpu_supports_avx2 = lambda: True  # type: ignore[attr-defined]
        module.residual_dist = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "flashspec_asm_kernel", module)
        with pytest.warns(RuntimeWarning, match="ABI mismatch"):
            available, fn, reason = _resolve_asm_backend()
        assert available is False
        assert fn is None
        assert "ABI" in reason

    def test_no_avx2_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A matching ABI without AVX2 stays on PyTorch."""
        module = types.ModuleType("flashspec_asm_kernel")
        module.abi_version = lambda: 1  # type: ignore[attr-defined]
        module.cpu_supports_avx2 = lambda: False  # type: ignore[attr-defined]
        module.residual_dist = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "flashspec_asm_kernel", module)
        available, _fn, reason = _resolve_asm_backend()
        assert available is False
        assert "avx2" in reason

    def test_matching_abi_selects_kernel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ABI 1 and AVX2 select the provided callable."""
        module = types.ModuleType("flashspec_asm_kernel")

        def _residual(p: object, q: object, out: object = None) -> object:
            return out

        module.abi_version = lambda: 1  # type: ignore[attr-defined]
        module.cpu_supports_avx2 = lambda: True  # type: ignore[attr-defined]
        module.residual_dist = _residual  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "flashspec_asm_kernel", module)
        available, fn, reason = _resolve_asm_backend()
        assert available is True
        assert fn is _residual
        assert reason == "selected"

    def test_missing_shared_library_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PyPI 0.1.0 imports, then raises ImportError because the .so is absent."""
        module = types.ModuleType("flashspec_asm_kernel")

        def _abi() -> int:
            raise ImportError(
                "libflashspec_asm.so not found. "
                "Build it with `make` from the repo root, or install the wheel."
            )

        module.abi_version = _abi  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "flashspec_asm_kernel", module)
        available, fn, reason = _resolve_asm_backend()
        assert available is False
        assert fn is None
        assert "ImportError" in reason
        assert "libflashspec_asm.so" in reason

    def test_simulated_0_1_0_oserror_disables_asm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Release 0.1.0 could import, then ``abi_version()`` raised OSError.

        Resolution catches that error, so the FlashSpec import stays usable
        and the residual stays on PyTorch.
        """
        module = types.ModuleType("flashspec_asm_kernel")

        def _abi() -> int:
            raise OSError("shared library failed to load")

        module.abi_version = _abi  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "flashspec_asm_kernel", module)
        available, fn, reason = _resolve_asm_backend()
        assert available is False
        assert fn is None
        assert "OSError" in reason
        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", False)
        monkeypatch.setattr(rejection, "_asm_residual_dist", None)
        p, q = _finite_pair(batch=1, vocab=7)
        got = residual_distribution(p, q)
        ref = torch.clamp(p - q, min=0.0)
        ref = ref / ref.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-8)

    def test_active_backend_matches_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """active_residual_backend follows the module flag."""
        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", False)
        assert active_residual_backend() == "pytorch"
        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        assert active_residual_backend() == "asm"

    def test_host_avx2_is_bool(self) -> None:
        """Host detection returns a bool and does not load the kernel."""
        assert isinstance(cpu_has_avx2(), bool)

    def test_real_kernel_nan_contract_when_selected(self) -> None:
        """When the real kernel is selected, NaN clamps to zero.

        The AVX2 ``vmaxps`` instruction returns the second operand on a
        quiet NaN. PyTorch preserves NaN. This assertion runs only on a
        machine where the shared library was selected.
        """
        if active_residual_backend() != "asm":
            pytest.skip(
                "flashspec-asm-kernel is not selected; PyTorch NaN behaviour is tested above"
            )
        p = torch.tensor([float("nan"), 0.5, 0.5])
        q = torch.zeros(3)
        got = residual_distribution(p, q)
        assert got[0].item() == 0.0
        assert torch.isfinite(got[1:]).all()


def _formula_matching_kernel(p: object, q: object, out: object) -> None:
    """Write ``max(0, p - q) / max(sum, 1e-9)`` into a float32 buffer."""
    import numpy as np

    pt = np.ascontiguousarray(p, dtype=np.float32)
    qt = np.ascontiguousarray(q, dtype=np.float32)
    residual = np.maximum(pt - qt, np.float32(0.0))
    denom = np.maximum(residual.sum(), np.float32(1e-9))
    written = residual / denom
    out[...] = written  # type: ignore[index]


def _torch_residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute the PyTorch residual without consulting the kernel flag."""
    residual = torch.clamp(p - q, min=0.0)
    return residual / residual.sum(dim=-1, keepdim=True).clamp(min=1e-9)


class TestKernelContract:
    """Routing, parity, and the NaN choice FlashSpec keeps."""

    def test_row_ready_requires_cpu_float32_contiguous_no_grad(self) -> None:
        """The kernel gate rejects grad, other dtypes, and non-contiguous rows."""
        ready = torch.zeros(4)
        assert _asm_row_ready(ready)
        assert not _asm_row_ready(ready.requires_grad_(True))
        assert not _asm_row_ready(torch.zeros(4, dtype=torch.float64))
        assert not _asm_row_ready(torch.zeros(4, dtype=torch.float16))
        assert not _asm_row_ready(torch.zeros(4, 2).t())

    def test_unready_output_row_uses_pytorch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A residual row that fails the kernel gate never calls the kernel."""
        calls = {"n": 0}

        def _count(p: object, q: object, out: object) -> None:
            calls["n"] += 1

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _count)
        monkeypatch.setattr(rejection, "_asm_row_ready", lambda _row: False)
        p, q = _finite_pair(batch=2, vocab=8)
        got = residual_distribution(p, q)
        assert calls["n"] == 0
        assert torch.allclose(got, _torch_residual(p, q), rtol=1e-5, atol=1e-8)

    def test_requires_grad_float32_still_reaches_kernel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Grad is detached before the kernel. ``.numpy()`` must not raise."""
        calls = {"n": 0}

        def _count(p: object, q: object, out: object) -> None:
            calls["n"] += 1
            _formula_matching_kernel(p, q, out)

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _count)
        p, q = _finite_pair(batch=1, vocab=8)
        p = p.requires_grad_(True)
        q = q.requires_grad_(True)
        got = residual_distribution(p, q)
        assert calls["n"] == 1
        assert torch.allclose(got, _torch_residual(p, q), rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16, torch.float16])
    def test_other_dtypes_never_reach_kernel(
        self, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
    ) -> None:
        """float64, bfloat16, and float16 stay on the PyTorch residual."""
        calls = {"n": 0}

        def _count(p: object, q: object, out: object) -> None:
            calls["n"] += 1

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _count)
        p, q = _finite_pair(batch=2, vocab=8)
        got = residual_distribution(p.to(dtype), q.to(dtype))
        assert calls["n"] == 0
        assert got.dtype == dtype

    def test_float32_reaches_kernel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same shapes in float32 do call the kernel."""
        calls = {"n": 0}

        def _count(p: object, q: object, out: object) -> None:
            calls["n"] += 1
            _formula_matching_kernel(p, q, out)

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _count)
        p, q = _finite_pair(batch=4, vocab=8)
        residual_distribution(p, q)
        assert calls["n"] == 4

    @pytest.mark.parametrize("batch", [1, 4])
    @pytest.mark.parametrize("vocab", [1, 7, 17, 32000, 128256])
    def test_parity_against_torch(
        self, monkeypatch: pytest.MonkeyPatch, vocab: int, batch: int
    ) -> None:
        """A formula-matching kernel matches PyTorch at rtol 1e-5, atol 1e-8.

        atol 1e-5 is too loose at vocab 128256: most residual entries are
        smaller than that. Vocab 1 is a probability vector ``[1]`` against
        ``[1]``, so the residual is identically 0 and the 1e-9 floor keeps
        the row sum at 0. For every larger vocab the row sums are 1 within
        1e-5. The real Linux wheel is not loaded in this process; this stub
        implements the same formula.
        """
        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _formula_matching_kernel)
        generator = torch.Generator().manual_seed(0)
        p = torch.softmax(torch.randn(batch, vocab, generator=generator), dim=-1)
        q = torch.softmax(torch.randn(batch, vocab, generator=generator), dim=-1)
        got = residual_distribution(p, q)
        ref = _torch_residual(p, q)
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-8)
        row_sum = got.sum(dim=-1)
        if vocab == 1:
            assert torch.allclose(row_sum, torch.zeros(batch), atol=1e-5)
        else:
            assert torch.allclose(row_sum, torch.ones(batch), atol=1e-5)

    def test_selected_kernel_nan_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FlashSpec keeps the kernel's NaN result. The documented kernel writes 0.

        PyTorch preserves NaN when the kernel is not selected. That case is
        ``TestPytorchResidual.test_nan_is_preserved_by_pytorch``. With the
        kernel selected, this function does not put the NaN back.
        ``ASM_NAN_CLAMP_TO_ZERO`` records that choice.
        """

        def _zero_nan(p: object, q: object, out: object) -> None:
            import numpy as np

            pt = np.array(p, dtype=np.float32, copy=True)
            qt = np.array(q, dtype=np.float32, copy=True)
            pt = np.where(np.isnan(pt), np.float32(0.0), pt)
            qt = np.where(np.isnan(qt), np.float32(0.0), qt)
            residual = np.maximum(pt - qt, np.float32(0.0))
            denom = np.maximum(residual.sum(), np.float32(1e-9))
            out[...] = residual / denom  # type: ignore[index]

        monkeypatch.setattr(rejection, "_ASM_AVAILABLE", True)
        monkeypatch.setattr(rejection, "_asm_residual_dist", _zero_nan)
        p = torch.tensor([float("nan"), 0.5, 0.5])
        q = torch.zeros(3)
        got = residual_distribution(p, q)
        assert ASM_NAN_CLAMP_TO_ZERO is True
        assert got[0].item() == 0.0
        assert torch.allclose(got[1:], torch.tensor([0.5, 0.5]))
