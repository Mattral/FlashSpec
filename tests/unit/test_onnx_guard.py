"""ONNX export guards. These tests do not write a model file."""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from flashspec.export.onnx import export_draft_to_onnx


def _call(opset: int = 17, gamma: int = 1) -> None:
    export_draft_to_onnx(
        nn.Linear(4, 4),
        "unused.onnx",
        torch.zeros(1, 2, dtype=torch.long),
        gamma=gamma,
        vocab_size=4,
        opset_version=opset,
    )


def test_rejects_old_opset_and_gamma() -> None:
    """The opset and gamma checks happen before any export."""
    with pytest.raises(ValueError, match="opset_version"):
        _call(opset=16)
    with pytest.raises(ValueError, match="gamma"):
        _call(gamma=0)


def test_missing_onnx_is_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing onnx package becomes a RuntimeError that names the extra."""
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(RuntimeError, match="flashspec\\[onnx\\]"):
        _call()


def test_export_exception_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tracer failure is wrapped and does not escape as the original type."""
    fake = types.ModuleType("onnx")
    fake.load = lambda path: object()  # type: ignore[attr-defined]
    fake.checker = types.SimpleNamespace(check_model=lambda model: None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx", fake)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("trace failed")

    monkeypatch.setattr(torch.onnx, "export", _boom)
    with pytest.raises(RuntimeError, match="ONNX export failed"):
        _call()
