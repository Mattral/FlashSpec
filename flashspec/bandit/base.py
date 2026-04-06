"""Abstract base class for online bandit draft selectors.

All concrete selectors (UCB1, Thompson, Oracle) inherit from ``DraftSelector``
and must honour its JSON serialisation and thread-safety contracts.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DraftSelector", "ArmStats"]

# ── Value object for per-arm statistics ───────────────────────────────────────


@dataclass(slots=True, frozen=False)
class ArmStats:
    """Mutable per-arm statistics used by bandit selectors.

    Parameters
    ----------
    n_pulls : int
        Total number of times this arm has been selected.
    n_accepted : int
        Total number of accepted tokens attributed to this arm.
    window_accepts : deque[int]
        Rolling window of per-round accept counts (1 or 0) for windowed stats.
    window_size : int
        Maximum size of the rolling window.  0 disables windowing.
    """

    n_pulls: int = 0
    n_accepted: int = 0
    window_accepts: deque[int] = field(default_factory=deque)
    window_size: int = 500

    def record(self, accepted: int) -> None:
        """Record the outcome of one round for this arm.

        Parameters
        ----------
        accepted : int
            Number of tokens accepted in this round (typically 0 or 1).

        Returns
        -------
        None

        Notes
        -----
        When ``window_size > 0`` the oldest entry is evicted once the window
        is full, so ``mean_accept_rate`` reflects only the most recent
        ``window_size`` rounds.

        Examples
        --------
        >>> stats = ArmStats(window_size=100)
        >>> stats.record(accepted=1)
        >>> stats.n_pulls
        1
        """
        self.n_pulls += 1
        self.n_accepted += accepted
        if self.window_size > 0:
            self.window_accepts.append(accepted)
            if len(self.window_accepts) > self.window_size:
                self.window_accepts.popleft()

    @property
    def mean_accept_rate(self) -> float:
        """Mean acceptance rate, optionally windowed.

        Returns
        -------
        float
            Windowed mean if ``window_size > 0`` and there are observations,
            else global mean, else 0.0.

        Notes
        -----
        When windowing is enabled (``window_size > 0``) the rate reflects
        only the last ``window_size`` rounds, allowing the bandit to track
        non-stationary acceptance distributions.

        Examples
        --------
        >>> stats = ArmStats(window_size=0)
        >>> stats.record(1); stats.record(0)
        >>> stats.mean_accept_rate
        0.5
        """
        if self.window_size > 0 and self.window_accepts:
            return sum(self.window_accepts) / len(self.window_accepts)
        if self.n_pulls > 0:
            return self.n_accepted / self.n_pulls
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys ``n_pulls``, ``n_accepted``,
            ``window_accepts``, and ``window_size``.

        Notes
        -----
        The returned dict can be passed directly to :meth:`from_dict` to
        reconstruct an identical ``ArmStats`` instance.

        Examples
        --------
        >>> stats = ArmStats(n_pulls=5, n_accepted=3, window_size=10)
        >>> d = stats.to_dict()
        >>> d["n_pulls"]
        5
        """
        return {
            "n_pulls": self.n_pulls,
            "n_accepted": self.n_accepted,
            "window_accepts": list(self.window_accepts),
            "window_size": self.window_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArmStats":
        """Deserialise from a dict produced by :meth:`to_dict`.

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary as returned by :meth:`to_dict`.

        Returns
        -------
        ArmStats
            Reconstructed instance with identical statistics.

        Notes
        -----
        The ``window_accepts`` deque is reconstructed with the original
        ``window_size`` as its ``maxlen``.

        Examples
        --------
        >>> stats = ArmStats(window_size=50)
        >>> stats.record(1)
        >>> restored = ArmStats.from_dict(stats.to_dict())
        >>> restored.n_pulls == stats.n_pulls
        True
        """
        obj = cls(
            n_pulls=d["n_pulls"],
            n_accepted=d["n_accepted"],
            window_size=d["window_size"],
        )
        obj.window_accepts = deque(
            d["window_accepts"], maxlen=d["window_size"] or None
        )
        return obj


# ── Abstract selector ─────────────────────────────────────────────────────────


class DraftSelector(ABC):
    """Abstract base class for online bandit draft-model selectors.

    Subclasses implement :meth:`select` and :meth:`update`.  All methods are
    thread-safe via a per-instance ``threading.Lock``.

    Parameters
    ----------
    n_arms : int
        Number of draft-model arms.
    window_size : int
        Rolling window size for acceptance statistics (0 = disabled).

    Raises
    ------
    ValueError
        If ``n_arms`` < 1 or ``window_size`` < 0.

    Notes
    -----
    The selector maintains one :class:`ArmStats` object per arm.
    The internal round counter ``t`` counts total calls to :meth:`update`.
    All public methods acquire ``self._lock`` before mutating state so that
    multiple generation workers can share a single selector safely.

    Examples
    --------
    >>> selector = UCB1Selector(n_arms=3, window_size=200)
    >>> arm = selector.select()
    >>> selector.update(arm, accepted=1)
    """

    def __init__(self, n_arms: int, window_size: int = 500) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1; got {n_arms}.")
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0; got {window_size}.")
        self._n_arms = n_arms
        self._window_size = window_size
        self._arms: list[ArmStats] = [
            ArmStats(window_size=window_size) for _ in range(n_arms)
        ]
        self._t: int = 0
        self._lock = threading.Lock()

    # ── Public interface ───────────────────────────────────────────────────

    @property
    def n_arms(self) -> int:
        """Number of arms.

        Returns
        -------
        int
            Count of available draft-model arms.

        Notes
        -----
        Fixed at construction time; cannot be changed after initialisation.

        Examples
        --------
        >>> selector = UCB1Selector(n_arms=3)
        >>> selector.n_arms
        3
        """
        return self._n_arms

    @property
    def t(self) -> int:
        """Total rounds elapsed (equal to the number of :meth:`update` calls).

        Returns
        -------
        int
            Non-negative integer round counter.

        Notes
        -----
        Resets to 0 after :meth:`reset` is called.

        Examples
        --------
        >>> selector = UCB1Selector(n_arms=2)
        >>> selector.update(0, accepted=1)
        >>> selector.t
        1
        """
        return self._t

    @abstractmethod
    def select(self) -> int:
        """Select an arm index to pull.

        Returns
        -------
        int
            Index in ``[0, n_arms)``.

        Notes
        -----
        Implementations must be thread-safe (acquire ``self._lock`` around
        any read-modify-write on shared state).

        Examples
        --------
        >>> arm = selector.select()
        >>> assert 0 <= arm < selector.n_arms
        """

    @abstractmethod
    def update(self, arm: int, accepted: int) -> None:
        """Record the outcome of pulling an arm.

        Parameters
        ----------
        arm : int
            Index of the arm that was pulled.
        accepted : int
            Number of tokens accepted in this round.

        Raises
        ------
        ValueError
            If ``arm`` is not in ``[0, n_arms)``.

        Notes
        -----
        Increments the internal round counter ``t`` and delegates to
        ``self._arms[arm].record(accepted)``.

        Examples
        --------
        >>> selector.update(0, accepted=1)
        """

    def reset(self) -> None:
        """Reset all arm statistics and the round counter to zero.

        Returns
        -------
        None

        Notes
        -----
        Intended for per-context-window resets when the prompt distribution
        shifts and accumulated statistics are no longer representative.
        Thread-safe: acquires ``self._lock`` before mutating state.

        Examples
        --------
        >>> selector.reset()
        >>> selector.t
        0
        """
        with self._lock:
            self._arms = [
                ArmStats(window_size=self._window_size)
                for _ in range(self._n_arms)
            ]
            self._t = 0

    def to_json(self) -> str:
        """Serialise bandit state to a JSON string.

        Returns
        -------
        str
            Compact JSON-encoded bandit state suitable for checkpointing.

        Notes
        -----
        Thread-safe: acquires ``self._lock`` before reading state.
        The returned string can be passed to :meth:`from_json` on any
        concrete subclass to reconstruct an identical instance.

        Examples
        --------
        >>> state_json = selector.to_json()
        >>> selector2 = UCB1Selector.from_json(state_json)
        """
        with self._lock:
            return json.dumps(self._state_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, json_str: str) -> "DraftSelector":
        """Restore bandit state from a JSON string produced by :meth:`to_json`.

        Parameters
        ----------
        json_str : str
            JSON string previously produced by :meth:`to_json`.

        Returns
        -------
        DraftSelector
            Restored selector instance with identical state.

        Raises
        ------
        ValueError
            If ``json_str`` is not valid JSON or is missing required fields.

        Notes
        -----
        Delegates to the concrete subclass's :meth:`_from_state_dict` method.
        The subclass is determined by the ``"type"`` key in the JSON object.

        Examples
        --------
        >>> json_str = selector.to_json()
        >>> restored = UCB1Selector.from_json(json_str)
        >>> restored.t == selector.t
        True
        """
        try:
            state = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for bandit state: {exc}") from exc
        return cls._from_state_dict(state)

    # ── Subclass hooks ─────────────────────────────────────────────────────

    @abstractmethod
    def _state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all state."""

    @classmethod
    @abstractmethod
    def _from_state_dict(cls, state: dict[str, Any]) -> "DraftSelector":
        """Restore an instance from a state dict."""
