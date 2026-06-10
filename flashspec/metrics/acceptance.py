"""Token acceptance rate tracker.

Tracks the running mean and per-step acceptance rate of draft tokens
across a generation session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AcceptanceTracker"]


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
    (α in the paper), averaged over all recorded steps.  It equals
    ``total_accepted / (step_count * gamma)``.

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

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``n_accepted`` is negative or exceeds ``gamma``.

        Notes
        -----
        Updates ``_total_accepted``, ``_total_possible`` (always incremented
        by ``gamma``), and ``_step_count`` atomically.  Thread safety is the
        caller's responsibility; the tracker itself does not lock.

        Examples
        --------
        >>> tracker = AcceptanceTracker(gamma=4)
        >>> tracker.record(n_accepted=2)
        >>> tracker.total_accepted
        2
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
        """Mean acceptance rate (α) across all recorded steps.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.  Returns 0.0 if no steps recorded.

        Notes
        -----
        Computed as ``total_accepted / total_possible`` where
        ``total_possible == step_count * gamma``.  This is the α used in
        the expected tokens-per-step formula: ``E[tokens/step] = γ·α + 1``.

        Examples
        --------
        >>> tracker = AcceptanceTracker(gamma=4)
        >>> tracker.record(3); tracker.record(1)
        >>> tracker.mean_acceptance_rate  # (3+1) / (4+4) = 0.5
        0.5
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
            Non-negative integer; 0 before any calls to :meth:`record`.

        Notes
        -----
        Each call to :meth:`record` increments this by exactly 1 regardless
        of ``n_accepted``.

        Examples
        --------
        >>> tracker = AcceptanceTracker(gamma=4)
        >>> tracker.record(2)
        >>> tracker.step_count
        1
        """
        return self._step_count

    @property
    def total_accepted(self) -> int:
        """Total draft tokens accepted across all steps.

        Returns
        -------
        int
            Cumulative sum of all ``n_accepted`` values passed to
            :meth:`record`.

        Notes
        -----
        Equals ``sum(n_accepted_i for all i)``.  Used together with
        ``step_count * gamma`` to compute ``mean_acceptance_rate``.

        Examples
        --------
        >>> tracker = AcceptanceTracker(gamma=4)
        >>> tracker.record(3); tracker.record(2)
        >>> tracker.total_accepted
        5
        """
        return self._total_accepted

    def reset(self) -> None:
        """Reset all counters to zero.

        Returns
        -------
        None

        Notes
        -----
        Intended for per-request or per-context-window resets so that
        acceptance statistics reflect only the current generation session.
        After calling this method, all properties return their initial values.

        Examples
        --------
        >>> tracker.reset()
        >>> tracker.mean_acceptance_rate
        0.0
        """
        self._total_accepted = 0
        self._total_possible = 0
        self._step_count = 0
