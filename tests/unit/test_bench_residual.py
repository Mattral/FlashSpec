"""Residual microbench document shape. Timers are stubbed."""

from __future__ import annotations

import pytest

import benchmarks.bench_residual as bench


def test_run_document_has_timestamp_and_e2e_unmeasured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A residual document can pass the result schema without e2e numbers."""
    sample = {"mean_ms": 0.1, "std_ms": 0.0, "iterations": 1.0}
    monkeypatch.setattr(bench, "_pytorch_residual_ms", lambda *args, **kwargs: sample)
    monkeypatch.setattr(bench, "_verification_ms", lambda *args, **kwargs: sample)
    monkeypatch.setattr(bench, "_kernel_ms", lambda *args, **kwargs: None)
    document = bench.run([8], iterations=1, warmup=0)
    assert document["status"] == "measured"
    assert isinstance(document["timestamp"], str) and document["timestamp"]
    for key in bench._E2E_UNMEASURED:
        assert key in document["unmeasured"]
    assert "asm_residual_v8" in document["unmeasured"]
    assert "rows" in document["metrics"]
