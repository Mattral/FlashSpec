"""Per-step time breakdown for the speculative loop.

Each step is split into four phases: drafting, target verification,
accept/reject sampling, and bookkeeping (appending tokens and moving the
KV caches). On CUDA the boundaries are ``torch.cuda.Event`` markers, so
profiling adds no host sync inside the loop; the events are resolved
once, when :meth:`StepProfiler.records` is read. On CPU the boundaries
are ``time.perf_counter`` readings.

The profiler is off by default. Benchmarks turn it on through
``MetricsConfig.profile_steps``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["PHASES", "StepProfiler", "StepRecord", "summarise_steps"]

PHASES: tuple[str, ...] = ("draft", "verify", "sample", "bookkeeping")


@dataclass(slots=True, frozen=True)
class StepRecord:
    """Time spent in each phase of one speculative step.

    Parameters
    ----------
    draft_ms : float
        Drafter time, including its forwards and draft sampling.
    verify_ms : float
        Target scoring time.
    sample_ms : float
        Accept/reject test, residual and bonus sampling.
    bookkeeping_ms : float
        Appending tokens, KV cache crop or catch-up, bandit and metric updates.
    tokens_emitted : int
        Tokens appended to the output by this step, summed over the rows
        that ran.
    """

    draft_ms: float
    verify_ms: float
    sample_ms: float
    bookkeeping_ms: float
    tokens_emitted: int

    @property
    def total_ms(self) -> float:
        """Sum of the four phases, in milliseconds."""
        return self.draft_ms + self.verify_ms + self.sample_ms + self.bookkeeping_ms


class StepProfiler:
    """Collect :class:`StepRecord` entries for one or more generations.

    Parameters
    ----------
    device : torch.device or str
        Device the models run on. CUDA uses events; anything else uses
        ``perf_counter``.

    Notes
    -----
    Call :meth:`start_step`, then :meth:`mark` at the end of each phase
    in order, then :meth:`end_step`. A phase that is not marked counts
    as zero. Marking a phase twice in one step adds both intervals.

    Examples
    --------
    >>> profiler = StepProfiler("cpu")
    >>> profiler.start_step()
    >>> profiler.mark("draft")
    >>> profiler.end_step(tokens_emitted=1)
    >>> len(profiler.records())
    1
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        resolved = torch.device(device)
        self._cuda = resolved.type == "cuda" and torch.cuda.is_available()
        self._pending: list[tuple[Any, list[tuple[str, Any]], int]] = []
        self._open: tuple[Any, list[tuple[str, Any]]] | None = None
        self._resolved: list[StepRecord] = []

    def _now(self) -> Any:  # noqa: ANN401 — float on CPU, cuda.Event on GPU
        if self._cuda:  # pragma: no cover - needs a CUDA device
            event_cls: Any = torch.cuda.Event
            event = event_cls(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def reset(self) -> None:
        """Drop every recorded step."""
        self._pending = []
        self._open = None
        self._resolved = []

    def start_step(self) -> None:
        """Open a new step at the current time."""
        self._open = (self._now(), [])

    def mark(self, phase: str) -> None:
        """Close ``phase`` at the current time.

        Parameters
        ----------
        phase : str
            One of :data:`PHASES`.

        Raises
        ------
        ValueError
            If ``phase`` is unknown.
        RuntimeError
            If no step is open.
        """
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}; expected one of {PHASES}.")
        if self._open is None:
            raise RuntimeError("mark() called before start_step().")
        self._open[1].append((phase, self._now()))

    def end_step(self, tokens_emitted: int) -> None:
        """Close the open step.

        Parameters
        ----------
        tokens_emitted : int
            Tokens this step appended to the output.

        Raises
        ------
        RuntimeError
            If no step is open.
        """
        if self._open is None:
            raise RuntimeError("end_step() called before start_step().")
        start, marks = self._open
        self._pending.append((start, marks, int(tokens_emitted)))
        self._open = None

    def _elapsed_ms(self, start: Any, end: Any) -> float:  # noqa: ANN401
        if self._cuda:  # pragma: no cover - needs a CUDA device
            return float(start.elapsed_time(end))
        return (float(end) - float(start)) * 1000.0

    def records(self) -> list[StepRecord]:
        """Return every closed step, resolving CUDA events if needed.

        Returns
        -------
        list[StepRecord]
            One entry per :meth:`end_step` call since the last reset.
        """
        if self._pending:
            if self._cuda:  # pragma: no cover - needs a CUDA device
                torch.cuda.synchronize()
            for start, marks, tokens in self._pending:
                spent = dict.fromkeys(PHASES, 0.0)
                previous = start
                for phase, marker in marks:
                    spent[phase] += self._elapsed_ms(previous, marker)
                    previous = marker
                self._resolved.append(
                    StepRecord(
                        draft_ms=spent["draft"],
                        verify_ms=spent["verify"],
                        sample_ms=spent["sample"],
                        bookkeeping_ms=spent["bookkeeping"],
                        tokens_emitted=tokens,
                    )
                )
            self._pending = []
        return list(self._resolved)

    def summary(self) -> dict[str, float]:
        """Aggregate the recorded steps. See :func:`summarise_steps`."""
        return summarise_steps(self.records())


def summarise_steps(steps: Sequence[StepRecord]) -> dict[str, float]:
    """Aggregate step records, for one generation or many.

    Parameters
    ----------
    steps : sequence of StepRecord
        Records to aggregate.

    Returns
    -------
    dict[str, float]
        ``n_steps``, ``tokens_emitted``, ``tokens_per_step_mean``,
        ``step_ms_mean``, and for each phase ``<phase>_ms_total``,
        ``<phase>_ms_mean`` and ``<phase>_fraction`` of the summed step
        time. Empty when ``steps`` is empty.
    """
    if not steps:
        return {}
    n_steps = len(steps)
    tokens = sum(step.tokens_emitted for step in steps)
    totals = {
        "draft": sum(step.draft_ms for step in steps),
        "verify": sum(step.verify_ms for step in steps),
        "sample": sum(step.sample_ms for step in steps),
        "bookkeeping": sum(step.bookkeeping_ms for step in steps),
    }
    grand = sum(totals.values())
    out: dict[str, float] = {
        "n_steps": float(n_steps),
        "tokens_emitted": float(tokens),
        "tokens_per_step_mean": tokens / n_steps,
        "step_ms_mean": grand / n_steps,
    }
    for phase in PHASES:
        out[f"{phase}_ms_total"] = totals[phase]
        out[f"{phase}_ms_mean"] = totals[phase] / n_steps
        out[f"{phase}_fraction"] = totals[phase] / grand if grand > 0.0 else 0.0
    return out
