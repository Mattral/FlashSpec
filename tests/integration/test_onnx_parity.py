"""Integration tests: ONNX export parity (§2.4 contract).

The spec states (§2.4):
  max(|onnx_logits - pytorch_logits|) < 1e-5   (all batch sizes 1..32)

All tests are skipped if ``onnx`` or ``onnxruntime`` are not installed.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch


def _onnx_available() -> bool:
    """Return True if both onnx and onnxruntime are importable."""
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _onnx_available(), reason="onnx / onnxruntime not installed")
class TestOnnxParity:
    """§2.4 ONNX parity: max |onnx_logits - pytorch_logits| < 1e-5."""

    def test_onnx_parity_batch_sizes_1_to_8(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """ONNX export matches PyTorch logits within 1e-5 for batch_size 1..8 (§2.4)."""
        import onnxruntime as ort
        from flashspec.export.onnx import export_draft_to_onnx

        toy_model_cpu.eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "draft_b1_8.onnx")
            export_draft_to_onnx(
                drafter=toy_model_cpu,
                output_path=onnx_path,
                example_input_ids=example_ids,
                gamma=4,
                vocab_size=1000,
            )
            session = ort.InferenceSession(onnx_path)

            for batch_size in range(1, 9):
                ids = torch.randint(0, 1000, (batch_size, 16))
                with torch.no_grad():
                    pt_logits = toy_model_cpu(ids).logits.float().numpy()
                ort_out = session.run(None, {"input_ids": ids.numpy()})
                ort_logits = ort_out[0]
                max_diff = float(abs(pt_logits - ort_logits).max())
                assert max_diff < 1e-5, (
                    f"§2.4 ONNX parity FAILED at batch_size={batch_size}: "
                    f"max_diff={max_diff:.2e} ≥ 1e-5"
                )

    def test_onnx_parity_batch_sizes_9_to_32(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """Edge case: ONNX parity holds for larger batch sizes 9..32 (§2.4)."""
        import onnxruntime as ort
        from flashspec.export.onnx import export_draft_to_onnx

        toy_model_cpu.eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "draft_b9_32.onnx")
            export_draft_to_onnx(
                drafter=toy_model_cpu,
                output_path=onnx_path,
                example_input_ids=example_ids,
                gamma=4,
                vocab_size=1000,
            )
            session = ort.InferenceSession(onnx_path)

            for batch_size in range(9, 33):
                ids = torch.randint(0, 1000, (batch_size, 16))
                with torch.no_grad():
                    pt_logits = toy_model_cpu(ids).logits.float().numpy()
                ort_out = session.run(None, {"input_ids": ids.numpy()})
                ort_logits = ort_out[0]
                max_diff = float(abs(pt_logits - ort_logits).max())
                assert max_diff < 1e-5, (
                    f"§2.4 ONNX parity FAILED at batch_size={batch_size}: "
                    f"max_diff={max_diff:.2e} ≥ 1e-5"
                )

    def test_onnx_export_different_sequence_lengths(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """Edge case: ONNX dynamic axis allows variable sequence lengths (§2.4)."""
        import onnxruntime as ort
        from flashspec.export.onnx import export_draft_to_onnx

        toy_model_cpu.eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "draft_seq_var.onnx")
            export_draft_to_onnx(
                drafter=toy_model_cpu,
                output_path=onnx_path,
                example_input_ids=example_ids,
                gamma=4,
                vocab_size=1000,
            )
            session = ort.InferenceSession(onnx_path)

            for seq_len in [4, 8, 16, 32, 64]:
                ids = torch.randint(0, 1000, (1, seq_len))
                with torch.no_grad():
                    pt_logits = toy_model_cpu(ids).logits.float().numpy()
                ort_out = session.run(None, {"input_ids": ids.numpy()})
                ort_logits = ort_out[0]
                max_diff = float(abs(pt_logits - ort_logits).max())
                assert max_diff < 1e-5, (
                    f"§2.4 ONNX parity FAILED at seq_len={seq_len}: "
                    f"max_diff={max_diff:.2e} ≥ 1e-5"
                )

    def test_onnx_invalid_opset_raises_value_error(
        self, toy_model_cpu: torch.nn.Module
    ) -> None:
        """Edge case: opset < 17 raises ValueError with a clear message."""
        from flashspec.export.onnx import export_draft_to_onnx

        toy_model_cpu.eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="opset_version"):
                export_draft_to_onnx(
                    drafter=toy_model_cpu,
                    output_path=os.path.join(tmpdir, "bad.onnx"),
                    example_input_ids=example_ids,
                    gamma=4,
                    vocab_size=1000,
                    opset_version=11,  # below minimum of 17
                )
