"""UCB1 online bandit draft selector.

Implements the UCB1 algorithm (Auer et al., 2002) for adaptive draft-model
selection.  The expected cumulative regret satisfies O(√(K T log T)) as
required by AGENTS.md §2.3.
"""

from __future__ import annotations

import math
from typing import Any

from flashspec.bandit.base import ArmStats, DraftSelector
from flashspec.utils.logging import get_logger

__all__ = ["UCB1Selector"]

logger = get_logger(__name__)

# UCB1 exploration constant — matches the theoretical derivation.
_DEFAULT_EXPLORATION_CONSTANT: float = 1.0


class UCB1Selector(DraftSelector):
    """UCB1 bandit selector for adaptive draft-model selection.

    Selects the arm with the highest upper confidence bound:

    .. math::

        \\text{score}_k = \\hat{\\mu}_k + c \\sqrt{\\frac{2 \\log t}{n_k}}

    where :math:`\\hat{\\mu}_k` is the arm's empirical mean acceptance rate,
    :math:`t` is the total number of rounds, :math:`n_k` is the number of
    pulls for arm :math:`k`, and :math:`c` is the exploration constant.

    Arms that have not been pulled yet are always explored first (infinite
    upper bound).

    Parameters
    ----------
    n_arms : int
        Number of draft-model arms.
    window_size : int
        Rolling window for acceptance statistics.  0 disables windowing.
    exploration_constant : float
        Multiplier for the exploration bonus.  Default 1.0.

    Raises
    ------
    ValueError
        If ``n_arms`` < 1, ``window_size`` < 0, or ``exploration_constant`` <= 0.

    Notes
    -----
    Regret bound: ``E[R_T] ≤ O(√(K T log T))`` (Auer et al., 2002).
    Verified by ``tests/unit/test_bandit.py``.

    References
    ----------
    .. [1] Auer et al. (2002), "Finite-time Analysis of the Multiarmed
       Bandit Problem", Machine Learning 47(2-3):235-256.

    Examples
    --------
    >>> selector = UCB1Selector(n_arms=3)
    >>> arm = selector.select()
    >>> selector.update(arm, accepted=1)
    >>> selector.to_json()  # checkpoint
    '{"type":"ucb1",...}'
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 500,
        exploration_constant: float = _DEFAULT_EXPLORATION_CONSTANT,
    ) -> None:
        if exploration_constant <= 0.0:
            raise ValueError(
                f"exploration_constant must be > 0; got {exploration_constant}."
            )
        super().__init__(n_arms=n_arms, window_size=window_size)
        self._c = exploration_constant

    def select(self) -> int:
        """Select the arm with the highest UCB1 score.

        Returns
        -------
        int
            Arm index in ``[0, n_arms)``.  Unpulled arms are always chosen
            before pulled arms.

        Notes
        -----
        Unpulled arms have an implicit score of +∞ and are always explored
        before any pulled arm.  Once all arms have been pulled at least once,
        the arm with the maximum UCB1 score is returned.  The lock is held
        for the duration of the score computation.

        Examples
        --------
        >>> arm = selector.select()
        >>> assert 0 <= arm < selector.n_arms
        """
        with self._lock:
            # Always explore unpulled arms first.
            for k, arm in enumerate(self._arms):
                if arm.n_pulls == 0:
                    return k
            # All arms pulled at least once: use UCB1 scores.
            t = max(self._t, 1)
            scores = [
                arm.mean_accept_rate + self._c * math.sqrt(2.0 * math.log(t) / arm.n_pulls)
                for arm in self._arms
            ]
            best = int(max(range(self._n_arms), key=lambda k: scores[k]))
            return best

    def update(self, arm: int, accepted: int) -> None:
        """Record the outcome of pulling ``arm``.

        Parameters
        ----------
        arm : int
            Arm index that was pulled.
        accepted : int
            Number of tokens accepted (0 or 1 per round).

        Raises
        ------
        ValueError
            If ``arm`` is not in ``[0, n_arms)``.

        Notes
        -----
        Increments the global round counter ``t`` and delegates to
        ``self._arms[arm].record(accepted)``.  The lock is held for the
        duration of the mutation.

        Examples
        --------
        >>> selector.update(0, accepted=1)
        """
        if not (0 <= arm < self._n_arms):
            raise ValueError(f"arm must be in [0, {self._n_arms}); got {arm}.")
        with self._lock:
            self._arms[arm].record(accepted)
            self._t += 1
        logger.debug(
            "UCB1 update",
            extra={"arm": arm, "accepted": accepted, "t": self._t},
        )

    def _state_dict(self) -> dict[str, Any]:
        return {
            "type": "ucb1",
            "n_arms": self._n_arms,
            "window_size": self._window_size,
            "exploration_constant": self._c,
            "t": self._t,
            "arms": [a.to_dict() for a in self._arms],
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, Any]) -> "UCB1Selector":
        obj = cls(
            n_arms=state["n_arms"],
            window_size=state["window_size"],
            exploration_constant=state["exploration_constant"],
        )
        obj._t = state["t"]
        obj._arms = [ArmStats.from_dict(d) for d in state["arms"]]
        return obj
