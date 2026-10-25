"""Bandit arms: which drafter proposes, and how many tokens.

An arm is a drafter (by index into the engine's ``drafters``) and a
speculation length ``gamma``. Several arms may share one drafter: a model
drafter at gamma 2, 4 and 6 is one model with one KV cache and three
arms. The engine keeps one cache per drafter, not per arm.

A *composite* arm names a lookup drafter as ``drafter`` and a model
drafter as ``fallback``: on each step it uses the lookup proposal when one
exists and the fallback drafter otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ArmSpec", "ArmStep", "arm_label", "default_arms", "validate_arms"]


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One bandit arm.

    Parameters
    ----------
    drafter : int
        Index of the proposing drafter in the engine's ``drafters``.
    gamma : int
        Tokens proposed per step (a lookup drafter proposes up to this
        many). Must be ``>= 1``.
    label : str
        Name used in logs and results. Empty uses ``"arm<k>"``.
    fallback : int or None
        Composite arm only: index of the model drafter used on steps where
        the lookup drafter ``drafter`` has no proposal. The fallback drafts
        ``gamma`` tokens too.
    """

    drafter: int
    gamma: int
    label: str = ""
    fallback: int | None = None


@dataclass(frozen=True, slots=True)
class ArmStep:
    """What one speculative step did, for per-arm reports.

    Parameters
    ----------
    arm : int
        Bandit arm index that was selected.
    drafter : int
        Drafter that proposed (a composite arm's lookup or fallback).
    proposed : int
        Draft tokens proposed: the gamma, or a lookup's real length.
    accepted : int
        Draft tokens accepted.
    emitted : int
        Tokens appended this step (accepted plus the residual or bonus,
        cut at EOS or ``max_new_tokens``).
    seconds : float
        Step wall time, the denominator of the bandit reward.
    reward : float
        The reward passed to the bandit: ``emitted / seconds``.
    drafter_pending : int
        Tokens the model drafter forwarded in its first draft call: its own
        leftover plus whatever other arms committed while it was idle (the
        catch-up). 0 for lookup steps.
    """

    arm: int
    drafter: int
    proposed: int
    accepted: int
    emitted: int
    seconds: float
    reward: float
    drafter_pending: int = 0


def arm_label(arms: Sequence[ArmSpec], index: int) -> str:
    """Return the arm's label, or ``"arm<index>"`` when it has none."""
    return arms[index].label or f"arm{index}"


def default_arms(n_drafters: int, gamma: int) -> list[ArmSpec]:
    """One arm per drafter at one ``gamma``: the engine's behaviour before arms existed."""
    return [ArmSpec(drafter=index, gamma=gamma) for index in range(n_drafters)]


def validate_arms(
    arms: Sequence[ArmSpec], n_drafters: int, lookup: Sequence[bool] | None = None
) -> None:
    """Raise ``ValueError`` for a missing drafter, a gamma below 1 or a bad composite.

    Parameters
    ----------
    arms : sequence of ArmSpec
        Arms to check.
    n_drafters : int
        Number of drafters the engine holds.
    lookup : sequence of bool or None
        Per drafter, whether it is a lookup drafter. Needed to check
        composite arms; ``None`` rejects every composite arm.
    """
    if not arms:
        raise ValueError("At least one arm is required.")
    for index, arm in enumerate(arms):
        if not 0 <= arm.drafter < n_drafters:
            raise ValueError(f"arm {index} names drafter {arm.drafter}; have {n_drafters}.")
        if arm.gamma < 1:
            raise ValueError(f"arm {index} has gamma {arm.gamma}; must be >= 1.")
        if arm.fallback is None:
            continue
        if not 0 <= arm.fallback < n_drafters:
            raise ValueError(f"arm {index} names fallback {arm.fallback}; have {n_drafters}.")
        if lookup is None or not lookup[arm.drafter] or lookup[arm.fallback]:
            raise ValueError(
                f"arm {index}: a composite arm needs a lookup drafter and a model-drafter fallback."
            )
