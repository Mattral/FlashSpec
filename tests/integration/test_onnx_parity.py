"""Integration test: ONNX export parity.

Verifies that exported ONNX draft models produce logits within 1e-5
of PyTorch for batch sizes 1..8.  Requires onnx and onnxruntime packages.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch


@pytest.mark.skipif(
    not (
        pytest.importorskip("onnx", reason="onnx not installed") is not None
        and pytest.importorskip("onnxruntime", reason="onnxruntime not installed") is not None
    )
    if False else False,
    reason="ONNX not available",
)
