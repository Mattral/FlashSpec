"""Tokens-per-second and Model FLOPs Utilisation (MFU) tracker.

All timed regions must use ``torch.cuda.synchronize()`` before stopping
the clock to ensure accurate wall-clock measurement on CUDA devices.
"""

from __future__ import annotations

import time

import torch

__all__ = ["ThroughputTracker"]

# Reference peak FLOP/s for known GPUs (in TFLOP/s, bfloat16).
_PEAK_TFLOPS: dict[str, float] = {
    "NVIDIA H100 SXM5 80GB": 989.0,
    "NVIDIA H100 PCIe 80GB": 756.0,
    "NVIDIA A100 SXM4 80GB": 312.0,
    "NVIDIA A100 PCIe 80GB": 312.0,
    "NVIDIA A10G": 125.0,
}


class ThroughputTracker:
    """Track tokens-per-second and Model FLOPs Utilisation (MFU).

    Parameters
    ----------
    model_flops_per_token : float
        Estimated FLOPs consumed by the target model per generated token.
        Used to compute MFU.  Set to 0.0 to disable MFU tracking.

    Notes
    -----
    MFU is computed as:

    .. math::

        \\text{MFU} = \\frac{\\text{tokens\\_per\\_second} \\times
            \\text{model\\_flops\\_per\\_token}}{\\text{peak\\_flops}}

    Peak FLOPs are looked up from the ``_PEAK_TFLOPS`` table using the
    GPU name reported by ``torch.cuda.get_device_name(0)``.  If the GPU
    is not in the table, MFU returns 0.0.

    Examples
    --------
    >>> tracker = ThroughputTracker(model_flops_per_token=1.2e12)
    >>> tracker.start()
    >>> tracker.stop(n_tokens=128)
    >>> tracker.mean_tokens_per_second > 0
    True
    """

    def __init__(self, model_flops_per_token: float = 0.0) -> None:
        self._model_flops_per_token = model_flops_per_token
        self._total_tokens: int = 0
        self._total_wall_seconds: float = 0.0
        self._step_count: int = 0
        self._start_time: float | None = None
        self._peak_flops: float = self._detect_peak_flops()

    def start(self) -> None:
        """Start the wall-clock timer for a generation step.

        Returns
        -------
        None

        Notes
        -----
        Calls ``torch.cuda.synchronize()`` before recording the start time
        when CUDA is available, ensuring all previously issued GPU kernels
        have completed.  Must be paired with :meth:`stop`.

        Examples
        --------
        >>> tracker.start()
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(self, n_tokens: int) -> float:
        """Stop the timer, record ``n_tokens`` generated, and return step throughput.

        Parameters
        ----------
        n_tokens : int
            Number of tokens produced in this step.  Must be ≥ 0.

        Returns
        -------
        float
            Tokens per second for this step only.  Returns 0.0 if the
            elapsed wall time is zero (e.g. on a mocked clock).

        Raises
        ------
        RuntimeError
            If :meth:`start` was not called before :meth:`stop`.
        ValueError
            If ``n_tokens`` < 0.

        Notes
        -----
        Calls ``torch.cuda.synchronize()`` before reading the stop time when
        CUDA is available.  Accumulates into internal counters so that
        :attr:`mean_tokens_per_second` reflects the average over all steps
        since construction (or the last :meth:`reset`).

        Examples
        --------
        >>> tracker.start()
        >>> tokens_per_second = tracker.stop(n_tokens=5)
        """
        if self._start_time is None:
            raise RuntimeError(
                "ThroughputTracker.stop() called without a preceding start()."
            )
        if n_tokens < 0:
            raise ValueError(f"n_tokens must be >= 0; got {n_tokens}.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        self._total_tokens += n_tokens
        self._total_wall_seconds += elapsed
        self._step_count += 1
        return n_tokens / elapsed if elapsed > 0.0 else 0.0

    @property
    def mean_tokens_per_second(self) -> float:
        """Mean throughput across all recorded steps.

        Returns
        -------
        float
            Tokens per second averaged over all steps since construction
            or the last :meth:`reset`.  Returns 0.0 if no steps recorded.

        Notes
        -----
        Computed as ``total_tokens / total_wall_seconds``.  This is the
        primary throughput metric reported in benchmark result JSON files
        (§14) and compared against regression thresholds in §6.

        Examples
        --------
        >>> tracker.mean_tokens_per_second
        1450.3
        """
        if self._total_wall_seconds <= 0.0:
            return 0.0
        return self._total_tokens / self._total_wall_seconds

    @property
    def mfu(self) -> float:
        """Model FLOPs Utilisation as a fraction of GPU peak throughput.

        Returns
        -------
        float
            MFU in ``[0.0, 1.0+]`` (values > 1.0 indicate the peak table
            entry is stale).  Returns 0.0 if ``model_flops_per_token`` is
            0.0 or the GPU is not in the lookup table.

        Notes
        -----
        MFU = tokens_per_second × model_flops_per_token / peak_flops.
        The ``_PEAK_TFLOPS`` table lists H100 SXM5 at 989 TFLOP/s and
        A100 SXM4 at 312 TFLOP/s (bfloat16 tensor core).  Extend the table
        for new GPUs in ``_PEAK_TFLOPS`` at the top of this module.

        Examples
        --------
        >>> tracker.mfu
        0.42
        """
        if self._model_flops_per_token <= 0.0 or self._peak_flops <= 0.0:
            return 0.0
        return (
            self.mean_tokens_per_second
            * self._model_flops_per_token
        ) / self._peak_flops

    @property
    def step_count(self) -> int:
        """Number of steps recorded (number of :meth:`stop` calls).

        Returns
        -------
        int
            Non-negative integer; 0 before any steps are recorded.

        Notes
        -----
        Unlike :class:`LatencyTracker`, this tracker is unbounded — every
        call to :meth:`stop` increments ``step_count`` with no eviction.

        Examples
        --------
        >>> tracker = ThroughputTracker()
        >>> tracker.step_count
        0
        """
        return self._step_count

    def reset(self) -> None:
        """Reset all counters to zero.

        Returns
        -------
        None

        Notes
        -----
        Called at the start of the measurement window (after the 50 warm-up
        steps in §6) so that warm-up overhead is excluded from the reported
        throughput.  The ``model_flops_per_token`` and ``_peak_flops``
        values are preserved across resets.

        Examples
        --------
        >>> tracker.reset()
        >>> tracker.step_count
        0
        """
        self._total_tokens = 0
        self._total_wall_seconds = 0.0
        self._step_count = 0
        self._start_time = None

    @staticmethod
    def _detect_peak_flops() -> float:
        """Attempt to read peak TFLOP/s for the current CUDA device.

        Returns
        -------
        float
            Peak FLOP/s in absolute units (not TFLOP/s), or 0.0 if
            CUDA is unavailable or the GPU is not in the lookup table.
        """
        if not torch.cuda.is_available():
            return 0.0
        gpu_name = torch.cuda.get_device_name(0)
        tflops = _PEAK_TFLOPS.get(gpu_name, 0.0)
        return tflops * 1e12
