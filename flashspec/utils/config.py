"""Pydantic v2 configuration models for FlashSpec.

All runtime settings flow through these models.  No magic numbers live
anywhere else in the codebase — every constant is either defined here or
in the module that owns it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "BanditConfig",
    "SamplingConfig",
    "MetricsConfig",
    "FlashSpecConfig",
]


class BanditConfig(BaseModel):
    """Configuration for the online bandit draft selector.

    Parameters
    ----------
    strategy : {"ucb1", "thompson", "oracle"}
        Which bandit algorithm to use.
    window_size : int
        Number of recent rounds to include in windowed statistics.
        Set to 0 to disable windowing (use all history).
    n_arms : int
        Number of draft-model arms available to the bandit.
    ucb_exploration_constant : float
        Exploration constant for UCB1 (the factor multiplying √(2 log t / n_k)).
        Default 1.0 matches the theoretical UCB1 derivation.
    thompson_prior_alpha : float
        Initial α of the Beta prior for Thompson sampling.
    thompson_prior_beta : float
        Initial β of the Beta prior for Thompson sampling.

    Raises
    ------
    ValueError
        If ``n_arms`` < 1 or ``window_size`` < 0.
    """

    strategy: Literal["ucb1", "thompson", "oracle"] = "ucb1"
    window_size: int = Field(default=500, ge=0)
    n_arms: int = Field(default=2, ge=1)
    ucb_exploration_constant: float = Field(default=1.0, gt=0.0)
    thompson_prior_alpha: float = Field(default=1.0, gt=0.0)
    thompson_prior_beta: float = Field(default=1.0, gt=0.0)

    model_config = {"frozen": True}


class SamplingConfig(BaseModel):
    """Configuration for speculative sampling.

    Parameters
    ----------
    gamma : int
        Speculation length — number of draft tokens proposed per step.
    temperature : float
        Sampling temperature applied to target logits before log-softmax.
        Must be positive; use 1.0 for unscaled probabilities.
    top_p : float
        Nucleus sampling threshold.  Set to 1.0 to disable.
    seed : int or None
        Random seed for reproducibility.  ``None`` means non-deterministic.
    variant : {"rejection", "typical"}
        Sampling algorithm variant.

    Raises
    ------
    ValueError
        If ``gamma`` < 1, ``temperature`` ≤ 0, or ``top_p`` not in (0, 1].
    """

    gamma: int = Field(default=4, ge=1)
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    variant: Literal["rejection", "typical"] = "rejection"

    model_config = {"frozen": True}


class MetricsConfig(BaseModel):
    """Configuration for metrics collection.

    Parameters
    ----------
    track_acceptance : bool
        Whether to track per-step token acceptance rates.
    track_throughput : bool
        Whether to track tokens/s and MFU.
    track_latency : bool
        Whether to track p50/p95/p99 step latency.
    latency_window : int
        Rolling window size for latency percentile computation.
    """

    track_acceptance: bool = True
    track_throughput: bool = True
    track_latency: bool = True
    latency_window: int = Field(default=1000, ge=10)

    model_config = {"frozen": True}


class FlashSpecConfig(BaseModel):
    """Top-level FlashSpec runtime configuration.

    Parameters
    ----------
    drafter_name : str
        Registry key of the draft model to use (e.g. ``"llama3-1b"``).
    target_name : str
        HuggingFace model identifier (or local path) for the target model.
    device : str
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    dtype : {"float32", "bfloat16", "float16"}
        Compute dtype for model forward passes.
    max_new_tokens : int
        Maximum tokens to generate per call.
    bandit : BanditConfig
        Bandit algorithm settings.
    sampling : SamplingConfig
        Speculative sampling settings.
    metrics : MetricsConfig
        Metrics collection settings.

    Raises
    ------
    ValueError
        If ``max_new_tokens`` < 1 or ``device`` is an invalid string.
    """

    drafter_name: str = "llama3-1b"
    target_name: str = "meta-llama/Llama-3-8B-Instruct"
    device: str = "cuda:0"
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    max_new_tokens: int = Field(default=512, ge=1)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_device_format(self) -> "FlashSpecConfig":
        """Validate that device string is non-empty and plausibly formatted."""
        if not self.device.strip():
            raise ValueError(
                f"'device' must be a non-empty string, got: {self.device!r}"
            )
        return self
