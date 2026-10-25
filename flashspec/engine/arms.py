"""Bandit arms: which drafter proposes, and how many tokens.

An arm is a drafter (by index into the engine's ``drafters``) and a
speculation length ``gamma``. Several arms may share one drafter: a model
drafter at gamma 2, 4 and 6 is one model with one KV cache and three
arms. The engine keeps one cache per drafter, not per arm.

A *composite* arm names a lookup drafter as ``drafter`` and a model
drafter as ``fallback``: on each step it uses the lookup proposal when one
exists and the fallback drafter otherwise. The lookup proposes up to
``gamma`` tokens; the fallback drafts ``fallback_gamma`` tokens (``gamma``
when it is not set).

:data:`DRAFTER_STRATEGIES` names the strategies
:class:`flashspec.utils.config.FlashSpecConfig` can select with
``drafter_strategy``, and :func:`strategy_arms` turns one into arms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "COMPOSITE_STRATEGIES",
    "DRAFTER_STRATEGIES",
    "ArmSpec",
    "ArmStep",
    "arm_label",
    "default_arms",
    "strategy_arms",
    "validate_arms",
]

#: Composite strategies: name -> (lookup gamma, fallback gamma). These are
#: the two candidates for the 0.3.0 default drafter strategy (HANDOFF
#: proposal (a)): ``composite-g4`` is the Phase 6 composite arm, and
#: ``composite-g6-f2`` proposes up to 6 lookup tokens and drafts 2 with the
#: model when the lookup has no match.
COMPOSITE_STRATEGIES: dict[str, tuple[int, int]] = {
    "composite-g4": (4, 4),
    "composite-g6-f2": (6, 2),
}

#: Every value of ``FlashSpecConfig.drafter_strategy``. ``"bandit"`` is one
#: arm per drafter at ``sampling.gamma``, chosen online by the bandit
#: (experimental: it lost to the best fixed arm in the Phase 6 run).
DRAFTER_STRATEGIES: tuple[str, ...] = ("bandit", *COMPOSITE_STRATEGIES)


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
        the lookup drafter ``drafter`` has no proposal.
    fallback_gamma : int or None
        Composite arm only: tokens the fallback drafts per step. ``None``
        uses ``gamma``. Must be ``>= 1``.
    """

    drafter: int
    gamma: int
    label: str = ""
    fallback: int | None = None
    fallback_gamma: int | None = None

    def step_gamma(self, *, fallback: bool) -> int:
        """Gamma of a step: the fallback's on a fallback step, else ``gamma``."""
        if fallback and self.fallback_gamma is not None:
            return self.fallback_gamma
        return self.gamma

    @property
    def max_gamma(self) -> int:
        """The longest draft any step of this arm proposes."""
        return max(self.gamma, self.step_gamma(fallback=True))


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


def strategy_arms(strategy: str, lookup: Sequence[bool], gamma: int) -> list[ArmSpec]:
    """Arms for a ``FlashSpecConfig.drafter_strategy``.

    Parameters
    ----------
    strategy : str
        One of :data:`DRAFTER_STRATEGIES`.
    lookup : sequence of bool
        Per drafter, whether it is a lookup drafter.
    gamma : int
        ``sampling.gamma``; used by ``"bandit"`` only.

    Returns
    -------
    list of ArmSpec
        ``"bandit"``: one arm per drafter at ``gamma``. A composite
        strategy: one composite arm over the first lookup drafter and the
        first model drafter, at the strategy's two gammas.

    Raises
    ------
    ValueError
        For an unknown strategy, or a composite strategy without one lookup
        and one model drafter.
    """
    if strategy == "bandit":
        return default_arms(len(lookup), gamma)
    if strategy not in COMPOSITE_STRATEGIES:
        raise ValueError(f"Unknown drafter strategy {strategy!r}; use one of {DRAFTER_STRATEGIES}.")
    lookups = [index for index, flag in enumerate(lookup) if flag]
    models = [index for index, flag in enumerate(lookup) if not flag]
    if not lookups or not models:
        raise ValueError(
            f"drafter_strategy {strategy!r} needs a lookup drafter (NgramDrafter) and a model "
            "drafter."
        )
    lookup_gamma, fallback_gamma = COMPOSITE_STRATEGIES[strategy]
    return [
        ArmSpec(
            drafter=lookups[0],
            gamma=lookup_gamma,
            label=strategy,
            fallback=models[0],
            fallback_gamma=None if fallback_gamma == lookup_gamma else fallback_gamma,
        )
    ]


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
            if arm.fallback_gamma is not None:
                raise ValueError(f"arm {index} sets fallback_gamma but has no fallback.")
            continue
        if arm.fallback_gamma is not None and arm.fallback_gamma < 1:
            raise ValueError(f"arm {index} has fallback_gamma {arm.fallback_gamma}; must be >= 1.")
        if not 0 <= arm.fallback < n_drafters:
            raise ValueError(f"arm {index} names fallback {arm.fallback}; have {n_drafters}.")
        if lookup is None or not lookup[arm.drafter] or lookup[arm.fallback]:
            raise ValueError(
                f"arm {index}: a composite arm needs a lookup drafter and a model-drafter fallback."
            )
