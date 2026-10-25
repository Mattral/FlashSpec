"""Bandit arms: which drafter proposes, and how many tokens.

An arm is a drafter (by index into the engine's ``drafters``) and a
speculation length ``gamma``. Several arms may share one drafter: a model
drafter at gamma 2, 4 and 6 is one model with one KV cache and three
arms. The engine keeps one cache per drafter, not per arm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ArmSpec", "arm_label", "default_arms", "validate_arms"]


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
    """

    drafter: int
    gamma: int
    label: str = ""


def arm_label(arms: Sequence[ArmSpec], index: int) -> str:
    """Return the arm's label, or ``"arm<index>"`` when it has none."""
    return arms[index].label or f"arm{index}"


def default_arms(n_drafters: int, gamma: int) -> list[ArmSpec]:
    """One arm per drafter at one ``gamma``: the engine's behaviour before arms existed."""
    return [ArmSpec(drafter=index, gamma=gamma) for index in range(n_drafters)]


def validate_arms(arms: Sequence[ArmSpec], n_drafters: int) -> None:
    """Raise ``ValueError`` if an arm names a missing drafter or a gamma below 1.

    Parameters
    ----------
    arms : sequence of ArmSpec
        Arms to check.
    n_drafters : int
        Number of drafters the engine holds.
    """
    if not arms:
        raise ValueError("At least one arm is required.")
    for index, arm in enumerate(arms):
        if not 0 <= arm.drafter < n_drafters:
            raise ValueError(f"arm {index} names drafter {arm.drafter}; have {n_drafters}.")
        if arm.gamma < 1:
            raise ValueError(f"arm {index} has gamma {arm.gamma}; must be >= 1.")
