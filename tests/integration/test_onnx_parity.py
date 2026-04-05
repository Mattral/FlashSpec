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
class TestOnnxParity:
    """ONNX export must satisfy max(|onnx - pytorch|) < 1e-5."""

    def test_onnx_export_and_parity_batch1_to_8(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """ONNX logits match PyTorch logits for batch sizes 1..8."""
        pytest.importorskip("onnx")
        onnx_runtime = pytest.importorskip("onnxruntime")

        from flashspec.export.onnx import export_draft_to_onnx

        toy_model_cpu.eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "draft.onnx")
            export_draft_to_onnx(
                drafter=toy_model_cpu,
                output_path=onnx_path,
                example_input_ids=example_ids,
                gamma=4,
                vocab_size=1000,
            )

            session = onnx_runtime.InferenceSession(onnx_path)

            for batch_size in range(1, 9):
                ids = torch.randint(0, 1000, (batch_size, 16))
                with torch.no_grad():
                    pt_logits = toy_model_cpu(ids).logits.float().numpy()

                ort_out = session.run(
                    None, {"input_ids": ids.numpy()}
                )
                ort_logits = ort_out[0]

                max_diff = abs(pt_logits - ort_logits).max()
                assert max_diff < 1e-5, (
                    f"ONNX parity failed at batch_size={batch_size}: "
                    f"max_diff={max_diff:.2e} >= 1e-5"
                )
