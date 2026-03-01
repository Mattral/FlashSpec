"""Triton-optimised kernels sub-package.

All consumers of kernels must import from this module, not from the
individual kernel files directly.  This is enforced by import-linter.

Private implementation modules:
- ``verify_kernel``: token acceptance kernel.
- ``gather_kernel``: accepted-token gather kernel.
- ``_reference``: pure-PyTorch reference implementations (tests only).
"""

from flashspec.kernels.gather_kernel import gather_accepted
from flashspec.kernels.verify_kernel import verify_tokens

__all__ = [
    "gather_accepted",
    "verify_tokens",
]
