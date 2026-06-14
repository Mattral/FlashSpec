"""Triton-optimised kernels sub-package.

All consumers of kernels must import from this module, not from the
individual kernel files directly.  This is enforced by import-linter.

Private implementation modules:
- ``verify_kernel``: token acceptance kernel (requires Triton).
- ``gather_kernel``: accepted-token gather kernel (requires Triton).
- ``_reference``: pure-PyTorch reference implementations (cross-platform,
  used by tests and as the runtime fallback when Triton is unavailable).

Notes
-----
Triton has no official PyPI wheels for Windows or macOS.  On those
platforms (or any platform where ``triton`` is not installed),
``verify_tokens`` and ``gather_accepted`` are bound to wrappers that raise
a clear, actionable ``ImportError`` when called, rather than failing at
import time.  Use ``flashspec.kernels._reference`` directly for a
cross-platform pure-PyTorch implementation with identical numerics
(verified in ``tests/unit/test_verify_kernel.py``).

To enable the accelerated kernels, install on Linux with a CUDA-capable
GPU::

    pip install flashspec[gpu]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "gather_accepted",
    "verify_tokens",
    "TRITON_AVAILABLE",
]

if TYPE_CHECKING:
    import torch

try:
    from flashspec.kernels.gather_kernel import gather_accepted
    from flashspec.kernels.verify_kernel import verify_tokens

    TRITON_AVAILABLE: bool = True
except ImportError as _exc:  # pragma: no cover — exercised on non-Linux CI
    _IMPORT_ERROR = _exc
    TRITON_AVAILABLE = False

    def _triton_unavailable(*_args: Any, **_kwargs: Any) -> "torch.Tensor":
        """Raise an actionable error explaining how to get Triton kernels.

        Raises
        ------
        ImportError
            Always. Explains that Triton is not installed on this platform
            and points to the pure-PyTorch reference implementation.
        """
        raise ImportError(
            "flashspec's accelerated Triton kernels are not available "
            f"(original error: {_IMPORT_ERROR}).\n\n"
            "Triton ships official PyPI wheels for Linux only. To use the "
            "GPU-accelerated kernels, install on Linux with a CUDA-capable "
            "GPU via:\n\n    pip install flashspec[gpu]\n\n"
            "For a cross-platform pure-PyTorch implementation with "
            "identical numerics, use "
            "flashspec.kernels._reference.verify_tokens_reference and "
            "flashspec.kernels._reference.gather_accepted_reference instead."
        )

    verify_tokens = _triton_unavailable
    gather_accepted = _triton_unavailable
