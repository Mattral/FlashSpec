"""FlashSpec — Adaptive speculative-decoding inference engine.

Adaptive speculative-decoding inference engine with Triton-optimised
verification and online bandit draft selection.

Public API surface (AGENTS.md §13.2 — do not modify without explicit approval):

    flashspec.SpeculativeEngine
    flashspec.GenerationResult
    flashspec.FlashSpecConfig
    flashspec.BanditConfig
    flashspec.SamplingConfig
    flashspec.MetricsConfig
    flashspec.register          (draft model decorator)
    flashspec.get_drafter
    flashspec.list_drafters
    flashspec.TRITON_AVAILABLE  (bool — True only on Linux with triton installed)

References
----------
.. [1] Leviathan et al. (2023), "Fast Inference from Transformers via
   Speculative Decoding", arXiv:2211.17192.
.. [2] Myet (2025), "FlashSpec: Adaptive Speculative Decoding with Online
   Bandit Draft Selection and Triton-Optimised Verification".
"""

from flashspec.engine.drafter import get_drafter, list_drafters, register
from flashspec.engine.speculative import GenerationResult, SpeculativeEngine
from flashspec.kernels import TRITON_AVAILABLE
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig

__all__ = [
    "BanditConfig",
    "FlashSpecConfig",
    "GenerationResult",
    "MetricsConfig",
    "SamplingConfig",
    "SpeculativeEngine",
    "TRITON_AVAILABLE",
    "get_drafter",
    "list_drafters",
    "register",
]

__version__ = "0.1.4"
__author__ = "Min Htet Myet (Mattral)"
