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

    Examples
    --------
    >>> tracker = ThroughputTracker(model_flops_per_token=1.2e12)
    >>> tracker.start()
    >>> # ... generate tokens ...
    >>> tracker.stop(n_tokens=128)
    >>> tracker.mean_tokens_per_second
    1450.3
    """

    def __init__(self, model_flops_per_token: float = 0.0) -> None:
        self._model_flops_per_token = model_flops_per_token
        self._total_tokens: int = 0
        self._total_wall_seconds: float = 0.0
        self._step_count: int = 0
        self._start_time: float | None = None
        self._peak_flops: float = self._detect_peak_flops()

    # ── Timing interface ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the wall-clock timer for a generation step.

        Examples
        --------
        >>> tracker.start()
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(self, n_tokens: int) -> float:
        """Stop the timer and record ``n_tokens`` generated.

        Parameters
        ----------
        n_tokens : int
            Number of tokens produced in this step.

        Returns
        -------
        float
            Tokens per second for this step.

        Raises
        ------
        RuntimeError
            If :meth:`start` was not called before :meth:`stop`.
        ValueError
            If ``n_tokens`` < 0.

        Examples
        --------
        >>> tps = tracker.stop(n_tokens=5)
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

    # ── Aggregate metrics ─────────────────────────────────────────────────

    @property
    def mean_tokens_per_second(self) -> float:
        """Mean throughput across all recorded steps.

        Returns
        -------
        float
            Tokens per second.  Returns 0.0 if no steps recorded.

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
        """Model FLOPs Utilisation fraction.

        Returns
        -------
        float
            MFU in ``[0.0, 1.0+]``.  Returns 0.0 if ``model_flops_per_token``
            is 0.0 or no GPU peak is available.

        Examples
        --------
        >>> tracker.mfu
        0.42
        """
        if self._model_flops_per_token <= 0.0 or self._peak_flops <= 0.0:
            return 0.0
        return (self.mean_tokens_per_second * self._model_flops_per_token) / (
            self._peak_flops * 1e12
        )

    @property
    def step_count(self) -> int:
        """Number of steps recorded.

        Returns
        -------
        int
        """
        return self._step_count

    def reset(self) -> None:
        """Reset all counters.

        Examples
        --------
        >>> tracker.reset()
        """
        self._total_tokens = 0
        self._total_wall_seconds = 0.0
        self._step_count = 0
        self._start_time = None

    # ── Internal helpers ──────────────────────────────────────────────────

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
