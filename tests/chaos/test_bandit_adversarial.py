"""Chaos tests: bandit under adversarial acceptance rates.

Tests edge cases including 0% acceptance, 100% acceptance, and sudden
best/worst arm swaps.
"""

from __future__ import annotations

import numpy as np

from flashspec.bandit.ucb import UCB1Selector
from flashspec.utils.device import set_seed


class TestBanditAdversarial:
    """UCB1 bandit under adversarial conditions."""

    def test_zero_acceptance_rate_does_not_crash(self) -> None:
        """Bandit handles 0% acceptance on all arms without error."""
        selector = UCB1Selector(n_arms=3)
        for _ in range(200):
            arm = selector.select()
            selector.update(arm, accepted=0)
        assert selector.t == 200, f"Expected t=200, got {selector.t}"

    def test_full_acceptance_rate_does_not_crash(self) -> None:
        """Bandit handles 100% acceptance on all arms without error."""
        selector = UCB1Selector(n_arms=3)
        for _ in range(200):
            arm = selector.select()
            selector.update(arm, accepted=1)
        assert selector.t == 200, f"Expected t=200, got {selector.t}"

    def test_adversarial_swap_recovery_within_200_rounds(self) -> None:
        """UCB1 recovers the new optimal arm within 200 rounds after a swap."""
        set_seed(55)
        rng = np.random.default_rng(55)
        true_rates = [0.2, 0.85]
        selector = UCB1Selector(n_arms=2, window_size=100)

        # Phase 1: converge.
        for _ in range(500):
            arm = selector.select()
            accepted = int(rng.random() < true_rates[arm])
            selector.update(arm, accepted)

        # Confirm convergence: arm 1 should be dominant.
        assert selector._arms[1].n_pulls > selector._arms[0].n_pulls, (
            "Arm 1 should be dominant after phase 1."
        )

        # Swap rates.
        true_rates = [0.85, 0.2]
        picks: list[int] = []
        for _ in range(200):
            arm = selector.select()
            picks.append(arm)
            accepted = int(rng.random() < true_rates[arm])
            selector.update(arm, accepted)

        # Recovery is a property of the selections, not of the arms' windowed
        # means: once UCB1 switches, the old best arm is pulled only a few
        # times, so its per-arm window still holds pre-swap rounds (at seed 55
        # arm 1 keeps 0.81 from 3 phase-2 pulls). Check what the bandit picks.
        recent = picks[-50:].count(0)
        assert recent >= 45, (
            f"After swap, arm 0 was picked in {recent}/50 of the last rounds; "
            "expected UCB1 to have switched to the new best arm (>= 45/50)."
        )
        assert picks.count(0) > picks.count(1), (
            f"Phase-2 picks: arm0={picks.count(0)}, arm1={picks.count(1)}"
        )

    def test_single_arm_bandit_always_selects_arm_zero(self) -> None:
        """Single-arm bandit always selects arm 0."""
        selector = UCB1Selector(n_arms=1)
        for _ in range(50):
            arm = selector.select()
            assert arm == 0, f"Expected arm=0, got {arm}"
            selector.update(arm, accepted=1)

    def test_bandit_with_many_arms_does_not_error(self) -> None:
        """Bandit with K=20 arms runs without error for 2000 rounds."""
        set_seed(0)
        rng = np.random.default_rng(0)
        k = 20
        true_rates = [rng.random() for _ in range(k)]
        selector = UCB1Selector(n_arms=k, window_size=100)
        for _ in range(2000):
            arm = selector.select()
            accepted = int(rng.random() < true_rates[arm])
            selector.update(arm, accepted)
        assert selector.t == 2000, f"Expected t=2000, got {selector.t}"
