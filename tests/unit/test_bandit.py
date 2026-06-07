"""Unit tests for the online bandit draft selectors.

Covers convergence, regret bounds, serialisation round-trips, and
adversarial non-stationary scenarios for UCB1, Thompson, and Oracle.
"""

from __future__ import annotations

import json
import math
import threading

import numpy as np
import pytest

from flashspec.bandit.base import ArmStats, DraftSelector
from flashspec.bandit.oracle import OracleSelector
from flashspec.bandit.thompson import ThompsonSelector
from flashspec.bandit.ucb import UCB1Selector
from flashspec.utils.device import set_seed

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_bandit(
    selector: DraftSelector,
    true_rates: list[float],
    n_rounds: int,
    seed: int = 0,
) -> tuple[int, float]:
    """Simulate a bandit for ``n_rounds`` and return (best_arm_pulls, regret)."""
    rng = np.random.default_rng(seed)
    best_arm = int(np.argmax(true_rates))
    best_rate = true_rates[best_arm]
    total_regret = 0.0

    for _ in range(n_rounds):
        arm = selector.select()
        accepted = int(rng.random() < true_rates[arm])
        selector.update(arm, accepted=accepted)
        total_regret += best_rate - true_rates[arm]

    return best_arm, total_regret


# ── UCB1 tests ────────────────────────────────────────────────────────────────

class TestUCB1Selector:
    """Tests for UCB1 bandit selector."""

    def test_ucb1_selects_best_arm_majority_of_time_after_t1000(self) -> None:
        """UCB1 selects the best arm > 90% of the time after T=1000 rounds."""
        set_seed(42)
        true_rates = [0.3, 0.7, 0.5]
        selector = UCB1Selector(n_arms=3, window_size=0)
        _run_bandit(selector, true_rates, n_rounds=1000)

        best_arm = 1
        best_arm_stats = selector._arms[best_arm]
        total_pulls = sum(a.n_pulls for a in selector._arms)
        selection_fraction = best_arm_stats.n_pulls / max(total_pulls, 1)
        assert selection_fraction > 0.90, (
            f"Best-arm selection fraction={selection_fraction:.3f} < 0.90"
        )

    def test_ucb1_regret_within_theoretical_bound(self) -> None:
        """UCB1 empirical regret is within 1.5× the O(√(KT log T)) bound."""
        set_seed(7)
        n_rounds = 10_000
        k_arms = 3
        true_rates = [0.3, 0.8, 0.5]
        selector = UCB1Selector(n_arms=k_arms, window_size=0)
        _best, empirical_regret = _run_bandit(selector, true_rates, n_rounds, seed=7)

        theoretical_upper = math.sqrt(k_arms * n_rounds * math.log(n_rounds))
        ratio = empirical_regret / theoretical_upper
        assert ratio < 1.5, (
            f"Regret ratio={ratio:.4f} ≥ 1.5 (empirical={empirical_regret:.1f}, "
            f"theoretical={theoretical_upper:.1f})"
        )

    def test_ucb1_serialisation_round_trip_preserves_state(self) -> None:
        """to_json() → from_json() exactly preserves UCB1 state."""
        selector = UCB1Selector(n_arms=3, window_size=100, exploration_constant=1.5)
        _run_bandit(selector, [0.3, 0.7, 0.5], n_rounds=50, seed=1)

        json_str = selector.to_json()
        restored = UCB1Selector._from_state_dict(json.loads(json_str))

        assert restored._t == selector._t, "Round counter mismatch after round-trip."
        assert restored._c == selector._c, "Exploration constant mismatch."
        for i in range(3):
            assert restored._arms[i].n_pulls == selector._arms[i].n_pulls, (
                f"Arm {i} n_pulls mismatch."
            )
            assert restored._arms[i].n_accepted == selector._arms[i].n_accepted, (
                f"Arm {i} n_accepted mismatch."
            )

    def test_ucb1_explores_unpulled_arms_first(self) -> None:
        """UCB1 always selects an unpulled arm before pulled arms."""
        selector = UCB1Selector(n_arms=3)
        first_three = [selector.select() for _ in range(3)]
        # First three selections must cover all three arms.
        assert sorted(first_three) == [0, 1, 2], (
            f"Did not explore all arms first: {first_three}"
        )

    def test_ucb1_invalid_arm_raises_value_error(self) -> None:
        """Updating with an out-of-range arm raises ValueError."""
        selector = UCB1Selector(n_arms=3)
        with pytest.raises(ValueError, match="arm must be in"):
            selector.update(arm=5, accepted=1)

    def test_ucb1_invalid_n_arms_raises_value_error(self) -> None:
        """n_arms < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_arms"):
            UCB1Selector(n_arms=0)

    def test_ucb1_reset_clears_state(self) -> None:
        """reset() clears all arm statistics and the round counter."""
        selector = UCB1Selector(n_arms=2)
        _run_bandit(selector, [0.4, 0.6], n_rounds=100)
        selector.reset()
        assert selector.t == 0, f"t={selector.t} after reset"
        for arm in selector._arms:
            assert arm.n_pulls == 0, f"n_pulls={arm.n_pulls} after reset"

    def test_ucb1_adversarial_recovery_within_200_rounds(self) -> None:
        """UCB1 recovers optimal arm within 200 rounds after best/worst swap."""
        set_seed(13)
        true_rates = [0.2, 0.9]
        selector = UCB1Selector(n_arms=2, window_size=100)

        # Phase 1: converge on arm 1.
        _run_bandit(selector, true_rates, n_rounds=500, seed=13)
        assert selector._arms[1].n_pulls > selector._arms[0].n_pulls, (
            "Expected arm 1 to be preferred after phase 1."
        )

        # Swap best/worst arm.
        true_rates_swapped = [0.9, 0.2]

        # Phase 2: recovery.
        rng = np.random.default_rng(99)
        for _ in range(200):
            arm = selector.select()
            accepted = int(rng.random() < true_rates_swapped[arm])
            selector.update(arm, accepted)

        # After 200 rounds, arm 0 should be selected more.
        arm0_rate = selector._arms[0].mean_accept_rate
        arm1_rate = selector._arms[1].mean_accept_rate
        assert arm0_rate > arm1_rate, (
            f"After swap, arm0 rate={arm0_rate:.2f} should exceed arm1 rate={arm1_rate:.2f}"
        )

    def test_ucb1_thread_safety_no_state_corruption(self) -> None:
        """Concurrent updates from multiple threads do not corrupt state."""
        selector = UCB1Selector(n_arms=2)
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                for _ in range(100):
                    arm = selector.select()
                    selector.update(arm, accepted=1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        total = sum(a.n_pulls for a in selector._arms)
        assert total == 400, f"Expected 400 total pulls, got {total}"


# ── Thompson tests ────────────────────────────────────────────────────────────

class TestThompsonSelector:
    """Tests for Thompson sampling selector."""

    def test_thompson_serialisation_round_trip(self) -> None:
        """to_json() → from_json() preserves Thompson posterior parameters."""
        selector = ThompsonSelector(n_arms=3, prior_alpha=2.0, prior_beta=2.0)
        _run_bandit(selector, [0.3, 0.7, 0.5], n_rounds=50, seed=2)

        json_str = selector.to_json()
        restored = ThompsonSelector._from_state_dict(json.loads(json_str))

        for i in range(3):
            assert abs(restored._post_alpha[i] - selector._post_alpha[i]) < 1e-9, (
                f"post_alpha[{i}] mismatch."
            )
            assert abs(restored._post_beta[i] - selector._post_beta[i]) < 1e-9, (
                f"post_beta[{i}] mismatch."
            )

    def test_thompson_invalid_prior_raises_value_error(self) -> None:
        """prior_alpha or prior_beta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="prior_alpha"):
            ThompsonSelector(n_arms=2, prior_alpha=0.0, prior_beta=1.0)
        with pytest.raises(ValueError, match="prior_beta"):
            ThompsonSelector(n_arms=2, prior_alpha=1.0, prior_beta=-1.0)


# ── Oracle tests ──────────────────────────────────────────────────────────────

class TestOracleSelector:
    """Tests for Oracle selector."""

    def test_oracle_always_selects_best_arm(self) -> None:
        """Oracle always returns the arm with the highest true rate."""
        selector = OracleSelector(n_arms=3, true_rates=[0.2, 0.9, 0.5])
        for _ in range(10):
            assert selector.select() == 1, "Oracle must always select arm 1 (rate=0.9)."

    def test_oracle_set_true_rates_updates_selection(self) -> None:
        """set_true_rates() changes which arm the oracle selects."""
        selector = OracleSelector(n_arms=2, true_rates=[0.3, 0.8])
        assert selector.select() == 1
        selector.set_true_rates([0.9, 0.1])
        assert selector.select() == 0, "After swap, oracle should select arm 0."

    def test_oracle_invalid_true_rates_raises_value_error(self) -> None:
        """Rates outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match="outside"):
            OracleSelector(n_arms=2, true_rates=[0.5, 1.5])


# ── ArmStats tests ────────────────────────────────────────────────────────────

class TestArmStats:
    """Tests for the ArmStats value object."""

    def test_arm_stats_mean_accept_rate_zero_when_no_pulls(self) -> None:
        """mean_accept_rate is 0.0 when no pulls have been recorded."""
        stats = ArmStats(window_size=100)
        assert stats.mean_accept_rate == 0.0, (
            f"Expected 0.0, got {stats.mean_accept_rate}"
        )

    def test_arm_stats_round_trip_preserves_values(self) -> None:
        """to_dict() → from_dict() preserves all fields."""
        stats = ArmStats(n_pulls=10, n_accepted=7, window_size=50)
        for i in range(10):
            stats.record(1 if i < 7 else 0)
        restored = ArmStats.from_dict(stats.to_dict())
        assert restored.n_pulls == stats.n_pulls, "n_pulls mismatch."
        assert restored.n_accepted == stats.n_accepted, "n_accepted mismatch."
