"""Count CUDA-graph captures, for the "observed N distinct sizes" question.

``torch.compile(mode="reduce-overhead")`` records one CUDA graph per
distinct input shape of a compiled function (cudagraph trees), and every
recording goes through ``torch.cuda.CUDAGraph.capture_begin``. Wrapping
that method counts recordings exactly, whoever triggers them. The count is
attributed to whatever ran while :meth:`CaptureCounter.section` was open.

On a machine without CUDA nothing is patched and every count stays 0;
results then say ``"available": false``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

__all__ = ["CaptureCounter"]


class CaptureCounter:
    """Process-wide CUDA-graph capture counter, split into named sections."""

    def __init__(self) -> None:
        self.available = bool(torch.cuda.is_available())
        self.total = 0
        self.by_section: dict[str, int] = {}
        self._current: str | None = None
        self._original: Any = None  # noqa: ANN401

    def install(self) -> CaptureCounter:
        """Patch ``CUDAGraph.capture_begin``. Safe to call twice; a no-op without CUDA."""
        if not self.available or self._original is not None:
            return self
        graph_cls: Any = torch.cuda.CUDAGraph
        original = graph_cls.capture_begin
        counter = self

        def _counted(graph: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            counter.total += 1
            if counter._current is not None:
                counter.by_section[counter._current] = (
                    counter.by_section.get(counter._current, 0) + 1
                )
            return original(graph, *args, **kwargs)

        graph_cls.capture_begin = _counted
        self._original = original
        return self

    def uninstall(self) -> None:
        """Restore ``CUDAGraph.capture_begin``."""
        if self._original is not None:
            graph_cls: Any = torch.cuda.CUDAGraph
            graph_cls.capture_begin = self._original
            self._original = None

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Attribute captures inside the block to ``name``."""
        previous = self._current
        self._current = name
        try:
            yield
        finally:
            self._current = previous

    def report(self, name: str) -> dict[str, object]:
        """JSON-ready count for one section."""
        return {
            "available": self.available,
            "captures": self.by_section.get(name, 0),
            "source": "torch.cuda.CUDAGraph.capture_begin calls while this method ran "
            "(warmup and measured generations); one per recorded CUDA graph",
        }
