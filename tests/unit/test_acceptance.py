"""Unit tests for the token acceptance rate tracker."""

from __future__ import annotations

import pytest

from flashspec.metrics.acceptance import AcceptanceTracker


class TestAcceptanceTracker:
    """Tests for AcceptanceTracker."""

    def test_mean_acceptance_rate_zero_when_no_steps_recorded(self) -> None:
        """mean_acceptance_rate is 0.0 when no steps have been recorded."""
        tracker = AcceptanceTracker(gamma=4)
        assert tracker.mean_acceptance_rate == 0.0, (
            f"Expected 0.0, got {tracker.mean_acceptance_rate}"
        )

    def test_mean_acceptance_rate_correct_after_single_step(self) -> None:
        """mean_acceptance_rate is correct after recording one step."""
        tracker = AcceptanceTracker(gamma=4)
        tracker.record(n_accepted=3)
        assert abs(tracker.mean_acceptance_rate - 0.75) < 1e-9, (
            f"Expected 0.75, got {tracker.mean_acceptance_rate}"
        )

    def test_mean_acceptance_rate_accumulates_correctly(self) -> None:
        """mean_acceptance_rate is correct after multiple steps."""
        tracker = AcceptanceTracker(gamma=4)
        tracker.record(n_accepted=4)  # 4/4
        tracker.record(n_accepted=2)  # 2/4
        # Total: 6 accepted / 8 possible = 0.75
        assert abs(tracker.mean_acceptance_rate - 0.75) < 1e-9, (
            f"Expected 0.75, got {tracker.mean_acceptance_rate}"
        )

    def test_all_accepted_rate_is_one(self) -> None:
        """mean_acceptance_rate is 1.0 when all tokens are accepted every step."""
        tracker = AcceptanceTracker(gamma=4)
        for _ in range(10):
            tracker.record(n_accepted=4)
        assert abs(tracker.mean_acceptance_rate - 1.0) < 1e-9, (
            f"Expected 1.0, got {tracker.mean_acceptance_rate}"
        )

    def test_zero_accepted_rate_is_zero(self) -> None:
        """mean_acceptance_rate is 0.0 when no tokens are accepted."""
        tracker = AcceptanceTracker(gamma=4)
        for _ in range(10):
            tracker.record(n_accepted=0)
        assert tracker.mean_acceptance_rate == 0.0, (
            f"Expected 0.0, got {tracker.mean_acceptance_rate}"
        )

    def test_invalid_n_accepted_above_gamma_raises_value_error(self) -> None:
        """n_accepted > gamma raises ValueError."""
        tracker = AcceptanceTracker(gamma=4)
        with pytest.raises(ValueError, match="n_accepted"):
            tracker.record(n_accepted=5)

    def test_invalid_n_accepted_negative_raises_value_error(self) -> None:
        """n_accepted < 0 raises ValueError."""
        tracker = AcceptanceTracker(gamma=4)
        with pytest.raises(ValueError, match="n_accepted"):
            tracker.record(n_accepted=-1)

    def test_step_count_increments_correctly(self) -> None:
        """step_count increments by 1 for each call to record()."""
        tracker = AcceptanceTracker(gamma=4)
        assert tracker.step_count == 0
        tracker.record(2)
        assert tracker.step_count == 1
        tracker.record(0)
        assert tracker.step_count == 2

    def test_reset_clears_all_counters(self) -> None:
        """reset() zeros out all counters."""
        tracker = AcceptanceTracker(gamma=4)
        tracker.record(n_accepted=3)
        tracker.reset()
        assert tracker.mean_acceptance_rate == 0.0, "Expected 0.0 after reset."
        assert tracker.step_count == 0, "Expected step_count=0 after reset."
        assert tracker.total_accepted == 0, "Expected total_accepted=0 after reset."
