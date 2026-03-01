"""Step latency tracker with p50/p95/p99 percentile computation.

Uses a fixed-size rolling window so memory usage is bounded regardless
of generation length.
"""

from __future__ import annotations

import time
from collections import deque

import torch

__all__ = ["LatencyTracker"]

_DEFAULT_WINDOW: int = 1000


class LatencyTracker:
    """Track per-step wall-clock latency and compute p50/p95/p99 percentiles.

    Parameters
    ----------
    window : int
        Number of recent steps to keep for percentile computation.
        Must be >= 10.

    Notes
    -----
    All timing uses ``time.perf_counter()`` with ``torch.cuda.synchronize()``
    bracketing the timed region on CUDA devices.

    Examples
    --------
    >>> tracker = LatencyTracker(window=500)
    >>> tracker.start()
    >>> # ... inference step ...
    >>> tracker.stop()
    >>> tracker.p99_ms
    47.8
    """

    def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
        if window < 10:
            raise ValueError(f"window must be >= 10; got {window}.")
        self._window = window
        self._latencies_ms: deque[float] = deque(maxlen=window)
        self._start_time: float | None = None

    def start(self) -> None:
        """Start the latency timer for one step.

        Examples
        --------
        >>> tracker.start()
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and record this step's latency.

        Returns
        -------
        float
            Latency of this step in milliseconds.

        Raises
        ------
        RuntimeError
            If :meth:`start` was not called before :meth:`stop`.

        Examples
        --------
        >>> latency_ms = tracker.stop()
        """
        if self._start_time is None:
            raise RuntimeError(
                "LatencyTracker.stop() called without a preceding start()."
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._start_time = None
        self._latencies_ms.append(elapsed_ms)
        return elapsed_ms

    def _percentile(self, pct: float) -> float:
        """Compute a percentile over the current window.

        Parameters
        ----------
        pct : float
            Percentile in ``[0, 100]``.

        Returns
        -------
        float
            Percentile value in milliseconds, or 0.0 if no data.
        """
        if not self._latencies_ms:
            return 0.0
        sorted_vals = sorted(self._latencies_ms)
        n = len(sorted_vals)
        idx = max(0, int(pct / 100.0 * n) - 1)
        return sorted_vals[min(idx, n - 1)]

    @property
    def p50_ms(self) -> float:
        """Median (p50) step latency in milliseconds.

        Returns
        -------
        float

        Examples
        --------
        >>> tracker.p50_ms
        38.1
        """
        return self._percentile(50.0)

    @property
    def p95_ms(self) -> float:
        """95th-percentile step latency in milliseconds.

        Returns
        -------
        float

        Examples
        --------
        >>> tracker.p95_ms
        44.2
        """
        return self._percentile(95.0)

    @property
    def p99_ms(self) -> float:
        """99th-percentile step latency in milliseconds.

        Returns
        -------
        float

        Examples
        --------
        >>> tracker.p99_ms
        47.8
        """
        return self._percentile(99.0)

    @property
    def step_count(self) -> int:
        """Number of latency samples recorded (up to window size).

        Returns
        -------
        int
        """
        return len(self._latencies_ms)

    def reset(self) -> None:
        """Clear all recorded latencies.

        Examples
        --------
        >>> tracker.reset()
        >>> tracker.step_count
        0
        """
        self._latencies_ms.clear()
        self._start_time = None
