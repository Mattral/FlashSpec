"""Oracle bandit selector — upper-bound baseline for regret experiments.

The Oracle always picks the arm with the highest *true* acceptance rate,
which must be supplied externally.  It is used only to compute the regret
upper bound in experiments; it is never used in production inference.
"""

from __future__ import annotations

from typing import Any

from flashspec.bandit.base import ArmStats, DraftSelector
from flashspec.utils.logging import get_logger

__all__ = ["OracleSelector"]

logger = get_logger(__name__)


class OracleSelector(DraftSelector):
    """Oracle bandit selector that always picks the true best arm.

    Requires ground-truth acceptance rates to be provided at construction
    time and updated via :meth:`set_true_rates`.  Used only in regret
    upper-bound experiments — never in production inference.

    Parameters
    ----------
    n_arms : int
        Number of draft-model arms.
    true_rates : list[float]
        Ground-truth acceptance rate for each arm.  Must have length ``n_arms``
        with values in ``[0, 1]``.
    window_size : int
        Rolling window for acceptance statistics.

    Raises
    ------
    ValueError
        If ``len(true_rates) != n_arms`` or any rate is outside ``[0, 1]``.

    Notes
    -----
    The oracle's cumulative reward serves as the upper bound for regret
    calculations in ``tests/unit/test_bandit.py``.

    Examples
    --------
    >>> selector = OracleSelector(n_arms=2, true_rates=[0.6, 0.9])
    >>> selector.select()
    1
    """

    def __init__(
        self,
        n_arms: int,
        true_rates: list[float],
        window_size: int = 500,
    ) -> None:
        if len(true_rates) != n_arms:
            raise ValueError(
                f"len(true_rates) must equal n_arms={n_arms}; "
                f"got {len(true_rates)}."
            )
        for i, r in enumerate(true_rates):
            if not (0.0 <= r <= 1.0):
                raise ValueError(
                    f"true_rates[{i}]={r} is outside [0, 1]."
                )
        super().__init__(n_arms=n_arms, window_size=window_size)
        self._true_rates: list[float] = list(true_rates)

    def select(self) -> int:
        """Return the arm index with the highest true acceptance rate.

        Returns
        -------
        int
            Arm index in ``[0, n_arms)``.

        Examples
        --------
        >>> OracleSelector(n_arms=2, true_rates=[0.4, 0.8]).select()
        1
        """
        with self._lock:
            return int(max(range(self._n_arms), key=lambda k: self._true_rates[k]))

    def update(self, arm: int, accepted: int) -> None:
        """Record outcome (used for regret tracking; does not affect selection).

        Parameters
        ----------
        arm : int
            Arm index that was pulled.
        accepted : int
            Number of accepted tokens.

        Raises
        ------
        ValueError
            If ``arm`` is not in ``[0, n_arms)``.

        Examples
        --------
        >>> selector.update(1, accepted=1)
        """
        if not (0 <= arm < self._n_arms):
            raise ValueError(f"arm must be in [0, {self._n_arms}); got {arm}.")
        with self._lock:
            self._arms[arm].record(accepted)
            self._t += 1

    def set_true_rates(self, true_rates: list[float]) -> None:
        """Update ground-truth acceptance rates (for non-stationary experiments).

        Parameters
        ----------
        true_rates : list[float]
            New ground-truth rates.  Must have the same length as ``n_arms``.

        Raises
        ------
        ValueError
            If length or values are invalid.

        Examples
        --------
        >>> selector.set_true_rates([0.9, 0.4])  # swap best/worst arm
        """
        if len(true_rates) != self._n_arms:
            raise ValueError(
                f"len(true_rates) must equal n_arms={self._n_arms}; "
                f"got {len(true_rates)}."
            )
        for i, r in enumerate(true_rates):
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"true_rates[{i}]={r} is outside [0, 1].")
        with self._lock:
            self._true_rates = list(true_rates)

    def _state_dict(self) -> dict[str, Any]:
        return {
            "type": "oracle",
            "n_arms": self._n_arms,
            "window_size": self._window_size,
            "true_rates": self._true_rates,
            "t": self._t,
            "arms": [a.to_dict() for a in self._arms],
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, Any]) -> "OracleSelector":
        obj = cls(
            n_arms=state["n_arms"],
            true_rates=state["true_rates"],
            window_size=state["window_size"],
        )
        obj._t = state["t"]
        obj._arms = [ArmStats.from_dict(d) for d in state["arms"]]
        return obj
