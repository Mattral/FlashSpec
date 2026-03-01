"""Token acceptance rate tracker.

Tracks the running mean and per-step acceptance rate of draft tokens
across a generation session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AcceptanceTracker"]

# Guard for division by zero.
_MIN_STEPS: int = 1


@dataclass(slots=True, frozen=False)
class AcceptanceTracker:
    """Track token acceptance rates across speculative decoding steps.

    Parameters
    ----------
    gamma : int
        Speculation length (max draft tokens per step).

    Notes
    -----
    ``mean_acceptance_rate`` is the fraction of draft tokens accepted
    (alpha in the paper), averaged over all recorded steps.

    Examples
    --------
    >>> tracker = AcceptanceTracker(gamma=4)
    >>> tracker.record(n_accepted=3)
    >>> tracker.mean_acceptance_rate
    0.75
    """

    gamma: int
    _total_accepted: int = field(default=0, init=False)
    _total_possible: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)

    def record(self, n_accepted: int) -> None:
        """Record the outcome of one speculative decoding step.

        Parameters
        ----------
        n_accepted : int
            Number of draft tokens accepted in this step.
            Must be in ``[0, gamma]``.

        Raises
        ------
        ValueError
            If ``n_accepted`` is negative or exceeds ``gamma``.

        Examples
        --------
        >>> tracker.record(n_accepted=2)
        """
        if not (0 <= n_accepted <= self.gamma):
            raise ValueError(
                f"n_accepted must be in [0, {self.gamma}]; got {n_accepted}."
            )
        self._total_accepted += n_accepted
        self._total_possible += self.gamma
        self._step_count += 1

    @property
    def mean_acceptance_rate(self) -> float:
        """Mean acceptance rate (alpha) across all recorded steps.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.  Returns 0.0 if no steps recorded.

        Examples
        --------
        >>> tracker.mean_acceptance_rate
        0.75
        """
        if self._total_possible == 0:
            return 0.0
        return self._total_accepted / self._total_possible

    @property
    def step_count(self) -> int:
        """Number of speculative steps recorded.

        Returns
        -------
        int
        """
        return self._step_count

    @property
    def total_accepted(self) -> int:
        """Total draft tokens accepted across all steps.

        Returns
        -------
        int
        """
        return self._total_accepted

    def reset(self) -> None:
        """Reset all counters to zero.

        Examples
        --------
        >>> tracker.reset()
        >>> tracker.mean_acceptance_rate
        0.0
        """
        self._total_accepted = 0
        self._total_possible = 0
        self._step_count = 0
