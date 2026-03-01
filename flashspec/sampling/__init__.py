"""Speculative sampling algorithms sub-package."""

from flashspec.sampling.rejection import rejection_sample
from flashspec.sampling.typical import typical_sample

__all__ = [
    "rejection_sample",
    "typical_sample",
]
