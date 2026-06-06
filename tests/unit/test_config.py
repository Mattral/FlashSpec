"""Unit tests for Pydantic v2 configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig


class TestSamplingConfig:
    """Tests for SamplingConfig validation."""

    def test_defaults_are_valid(self) -> None:
        """SamplingConfig default values pass validation."""
        cfg = SamplingConfig()
        assert cfg.gamma == 4
        assert cfg.temperature == 1.0

    def test_gamma_below_one_raises_validation_error(self) -> None:
        """gamma < 1 raises ValidationError."""
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
        assert cfg.strategy == "ucb1"
        assert cfg.n_arms == 2

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
    """Tests for FlashSpecConfig."""

    def test_defaults_construct_successfully(self) -> None:
        """Default FlashSpecConfig constructs without errors."""
        cfg = FlashSpecConfig()
        assert cfg.device == "cuda:0"
        assert cfg.max_new_tokens == 512

    def test_empty_device_raises_validation_error(self) -> None:
        """Empty device string raises ValidationError."""
        with pytest.raises(ValueError, match="device"):
            FlashSpecConfig(device="   ")

    def test_max_new_tokens_zero_raises_validation_error(self) -> None:
        """max_new_tokens < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            FlashSpecConfig(max_new_tokens=0)

    def test_nested_bandit_config_propagates(self) -> None:
        """Nested BanditConfig is accessible on the parent config."""
        cfg = FlashSpecConfig(bandit=BanditConfig(n_arms=4, strategy="thompson"))
        assert cfg.bandit.n_arms == 4
        assert cfg.bandit.strategy == "thompson"

    def test_config_is_immutable(self) -> None:
        """FlashSpecConfig fields cannot be mutated after construction."""
        cfg = FlashSpecConfig()
        with pytest.raises(Exception):
            cfg.device = "cpu"  # type: ignore[misc]
