"""Speculative sampling algorithms sub-package."""

from flashspec.sampling.greedy import greedy_verify
from flashspec.sampling.rejection import rejection_sample
from flashspec.sampling.typical import typical_sample

__all__ = [
    "greedy_verify",
    "rejection_sample",
    "typical_sample",
]
