"""Pydantic v2 configuration models for FlashSpec.

All runtime settings flow through these models.  No magic numbers live
anywhere else in the codebase — every constant is either defined here or
in the module that owns it.
"""

from __future__ import annotations

from typing import Any, Literal

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
        Use 1.0 for unscaled probabilities. ``0`` means greedy decoding and
        selects ``variant="greedy"``.
    top_p : float
        Nucleus sampling threshold.  Set to 1.0 to disable.
    seed : int or None
        Random seed for reproducibility.  ``None`` means non-deterministic.
    variant : {"rejection", "typical", "greedy"}
        Sampling algorithm variant. ``"greedy"`` accepts draft tokens up to
        the first one that differs from the target's argmax, so the output
        equals the target's own greedy decode. It requires
        ``temperature == 0`` and is chosen automatically for it.

    Raises
    ------
    ValueError
        If ``gamma`` < 1, ``temperature`` < 0, ``top_p`` not in (0, 1], or
        ``temperature == 0`` and ``variant`` disagree.
    """

    gamma: int = Field(default=4, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    variant: Literal["rejection", "typical", "greedy"] = "rejection"

    model_config = {"frozen": True}

    @model_validator(mode="before")
    @classmethod
    def _greedy_at_zero_temperature(cls, data: Any) -> Any:  # noqa: ANN401
        """Pick ``variant="greedy"`` for ``temperature == 0`` unless one was given."""
        if isinstance(data, dict) and data.get("temperature") == 0 and "variant" not in data:
            return {**data, "variant": "greedy"}
        return data

    @model_validator(mode="after")
    def _greedy_iff_zero_temperature(self) -> SamplingConfig:
        """Greedy decoding and ``temperature == 0`` go together."""
        if (self.variant == "greedy") != (self.temperature == 0.0):
            raise ValueError(
                f"variant='greedy' requires temperature=0 and temperature=0 requires "
                f"variant='greedy'; got variant={self.variant!r}, "
                f"temperature={self.temperature}."
            )
        return self

    @property
    def greedy(self) -> bool:
        """True for ``variant="greedy"``."""
        return self.variant == "greedy"


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
    profile_steps : bool
        Record a per-step draft/verify/sample/bookkeeping breakdown
        (:class:`flashspec.metrics.step_profile.StepProfiler`). Off by
        default; the benchmark harness turns it on.
    """

    track_acceptance: bool = True
    track_throughput: bool = True
    track_latency: bool = True
    latency_window: int = Field(default=1000, ge=10)
    profile_steps: bool = False

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
    eos_token_ids : tuple[int, ...]
        Token ids that end a sequence. Generation of a row stops right
        after it emits one of them; the stop token is kept. Empty (the
        default) always generates ``max_new_tokens``.
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
    eos_token_ids: tuple[int, ...] = ()
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_device_format(self) -> FlashSpecConfig:
        """Validate that device string is non-empty and plausibly formatted."""
        if not self.device.strip():
            raise ValueError(f"'device' must be a non-empty string, got: {self.device!r}")
        return self
