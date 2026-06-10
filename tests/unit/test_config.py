"""Unit tests for Pydantic v2 configuration models."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig


class TestSamplingConfig:
    """Tests for SamplingConfig validation."""

    def test_defaults_are_valid(self) -> None:
        """SamplingConfig default values pass validation."""
        cfg = SamplingConfig()
        assert cfg.gamma == 4, (
            f"Default gamma must be 4; got {cfg.gamma}"
        )
        assert cfg.temperature == 1.0, (
            f"Default temperature must be 1.0; got {cfg.temperature}"
        )

    def test_gamma_below_one_raises_validation_error(self) -> None:
        """gamma < 1 raises ValidationError with 'gamma' in the message."""
        with pytest.raises(ValidationError, match="gamma"):
            SamplingConfig(gamma=0)

    def test_temperature_zero_raises_validation_error(self) -> None:
        """temperature <= 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SamplingConfig(temperature=0.0)

    def test_top_p_above_one_raises_validation_error(self) -> None:
        """top_p > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SamplingConfig(top_p=1.1)


class TestBanditConfig:
    """Tests for BanditConfig validation."""

    def test_defaults_are_valid(self) -> None:
        """BanditConfig default values pass validation."""
        cfg = BanditConfig()
        assert cfg.strategy == "ucb1", (
            f"Default strategy must be 'ucb1'; got {cfg.strategy!r}"
        )
        assert cfg.n_arms == 2, (
            f"Default n_arms must be 2; got {cfg.n_arms}"
        )

    def test_n_arms_zero_raises_validation_error(self) -> None:
        """n_arms < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            BanditConfig(n_arms=0)

    def test_invalid_strategy_raises_validation_error(self) -> None:
        """Unknown strategy raises ValidationError."""
        with pytest.raises(ValidationError):
            BanditConfig(strategy="unknown_algorithm")  # type: ignore[arg-type]

    def test_negative_window_size_raises_validation_error(self) -> None:
        """window_size < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            BanditConfig(window_size=-1)


class TestFlashSpecConfig:
    """Tests for FlashSpecConfig top-level configuration."""

    def test_defaults_construct_successfully(self) -> None:
        """Default FlashSpecConfig constructs without errors."""
        cfg = FlashSpecConfig()
        assert cfg.device == "cuda:0", (
            f"Default device must be 'cuda:0'; got {cfg.device!r}"
        )
        assert cfg.max_new_tokens == 512, (
            f"Default max_new_tokens must be 512; got {cfg.max_new_tokens}"
        )

    def test_empty_device_raises_validation_error(self) -> None:
        """Empty device string raises ValueError."""
        with pytest.raises(ValueError, match="device"):
            FlashSpecConfig(device="   ")

    def test_max_new_tokens_zero_raises_validation_error(self) -> None:
        """max_new_tokens < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            FlashSpecConfig(max_new_tokens=0)

    def test_nested_bandit_config_propagates(self) -> None:
        """Nested BanditConfig fields are accessible on the parent config."""
        cfg = FlashSpecConfig(bandit=BanditConfig(n_arms=4, strategy="thompson"))
        assert cfg.bandit.n_arms == 4, (
            f"Expected nested bandit.n_arms=4; got {cfg.bandit.n_arms}"
        )
        assert cfg.bandit.strategy == "thompson", (
            f"Expected nested bandit.strategy='thompson'; got {cfg.bandit.strategy!r}"
        )

    def test_config_is_immutable(self) -> None:
        """FlashSpecConfig fields cannot be mutated after construction (frozen=True)."""
        cfg = FlashSpecConfig()
        with pytest.raises(Exception):
            cfg.device = "cpu"  # type: ignore[misc]


# ── Hypothesis property-based tests (§5.1) ───────────────────────────────────


@given(gamma=st.integers(min_value=1, max_value=32))
@settings(max_examples=30, deadline=None)
def test_sampling_config_valid_gamma_always_constructs(gamma: int) -> None:
    """Property: SamplingConfig accepts any gamma ≥ 1 without error."""
    cfg = SamplingConfig(gamma=gamma)
    assert cfg.gamma == gamma, (
        f"Expected gamma={gamma}, got {cfg.gamma}"
    )


@given(
    n_arms=st.integers(min_value=1, max_value=20),
    window_size=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=25, deadline=None)
def test_bandit_config_valid_n_arms_and_window_constructs(
    n_arms: int, window_size: int
) -> None:
    """Property: BanditConfig accepts any n_arms ≥ 1 and window_size ≥ 0."""
    cfg = BanditConfig(n_arms=n_arms, window_size=window_size)
    assert cfg.n_arms == n_arms, (
        f"Expected n_arms={n_arms}, got {cfg.n_arms}"
    )
    assert cfg.window_size == window_size, (
        f"Expected window_size={window_size}, got {cfg.window_size}"
    )


@given(max_new_tokens=st.integers(min_value=1, max_value=4096))
@settings(max_examples=20, deadline=None)
def test_flashspec_config_valid_max_new_tokens_constructs(
    max_new_tokens: int,
) -> None:
    """Property: FlashSpecConfig accepts any max_new_tokens ≥ 1."""
    cfg = FlashSpecConfig(max_new_tokens=max_new_tokens)
    assert cfg.max_new_tokens == max_new_tokens, (
        f"Expected max_new_tokens={max_new_tokens}, got {cfg.max_new_tokens}"
    )
