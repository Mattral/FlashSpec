"""``benchmarks.cuda_graphs.CaptureCounter``: counts capture_begin calls per section."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from benchmarks.cuda_graphs import CaptureCounter


class _FakeGraph:
    def __init__(self) -> None:
        self.begun = 0

    def capture_begin(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        self.begun += 1


def test_counts_by_section_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Captures inside a section are attributed to it; uninstall restores the method."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    original = _FakeGraph.capture_begin
    counter = CaptureCounter().install()
    graph = _FakeGraph()
    graph.capture_begin()
    with counter.section("a"):
        graph.capture_begin()
        graph.capture_begin(pool=None)
    with counter.section("b"):
        graph.capture_begin()
    assert graph.begun == 4
    assert counter.total == 4
    assert counter.by_section == {"a": 2, "b": 1}
    assert counter.report("a")["captures"] == 2
    assert counter.report("missing")["captures"] == 0
    counter.uninstall()
    assert _FakeGraph.capture_begin is original


def test_no_cuda_is_a_no_op() -> None:
    """Without CUDA nothing is patched and reports say unavailable."""
    if torch.cuda.is_available():  # pragma: no cover - CPU CI
        pytest.skip("CUDA present")
    counter = CaptureCounter().install()
    assert counter.report("x") == {
        "available": False,
        "captures": 0,
        "source": counter.report("x")["source"],
    }
    counter.uninstall()
