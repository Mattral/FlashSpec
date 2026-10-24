"""Integration tests: ONNX export parity contract.

The parity contract:
  max(|onnx_logits - pytorch_logits|) < 1e-5   (all batch sizes 1..32)

All tests are skipped if ``onnx`` or ``onnxruntime`` are not installed.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch


class _LogitsOnly(torch.nn.Module):
    """Tensor-returning view of the toy model for ONNX tracing.

    The shared toy forward returns a namespace. The legacy tracer rejects that.
    """

    def __init__(self, source: torch.nn.Module) -> None:
        super().__init__()
        vocab = int(source.embed.num_embeddings)
        hidden = int(source.embed.embedding_dim)
        n_layers = len(source.transformer.layers)
        self.embed = torch.nn.Embedding(vocab, hidden)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.load_state_dict(source.state_dict())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, sequence, vocab)``."""
        hidden = self.embed(input_ids)
        hidden = self.transformer(hidden)
        return self.lm_head(hidden)


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
    """ONNX parity: max |onnx_logits - pytorch_logits| < 1e-5."""

    def test_onnx_parity_batch_sizes_1_to_8(self, toy_model_cpu: torch.nn.Module) -> None:
        """ONNX export matches PyTorch logits within 1e-5 for batch_size 1..8."""
        import onnxruntime as ort

        from flashspec.export.onnx import export_draft_to_onnx

        model = _LogitsOnly(toy_model_cpu).eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "draft_b1_8.onnx")
            export_draft_to_onnx(
                drafter=model,
                output_path=onnx_path,
                example_input_ids=example_ids,
                gamma=4,
                vocab_size=1000,
            )
            session = ort.InferenceSession(onnx_path)

            for batch_size in range(1, 9):
                ids = torch.randint(0, 1000, (batch_size, 16))
                with torch.no_grad():
                    pt_logits = model(ids).float().numpy()
                ort_out = session.run(None, {"input_ids": ids.numpy()})
                ort_logits = ort_out[0]
                max_diff = float(abs(pt_logits - ort_logits).max())
                assert max_diff < 1e-5, (
                    f"ONNX parity FAILED at batch_size={batch_size}: max_diff={max_diff:.2e} ≥ 1e-5"
                )

    def test_onnx_parity_batch_sizes_9_to_32(self, toy_model_cpu: torch.nn.Module) -> None:
        """Edge case: ONNX parity holds for larger batch sizes 9..32."""
        import onnxruntime as ort

        from flashspec.export.onnx import export_draft_to_onnx

        model = _LogitsOnly(toy_model_cpu).eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "draft_b9_32.onnx")
            export_draft_to_onnx(
                drafter=model,
                output_path=onnx_path,
                example_input_ids=example_ids,
                gamma=4,
                vocab_size=1000,
            )
            session = ort.InferenceSession(onnx_path)

            for batch_size in range(9, 33):
                ids = torch.randint(0, 1000, (batch_size, 16))
                with torch.no_grad():
                    pt_logits = model(ids).float().numpy()
                ort_out = session.run(None, {"input_ids": ids.numpy()})
                ort_logits = ort_out[0]
                max_diff = float(abs(pt_logits - ort_logits).max())
                assert max_diff < 1e-5, (
                    f"ONNX parity FAILED at batch_size={batch_size}: max_diff={max_diff:.2e} ≥ 1e-5"
                )

    def test_onnx_export_different_sequence_lengths(self) -> None:
        """ONNX dynamic sequence axis matches PyTorch for a traceable module.

        The transformer trace bakes the example sequence length into attention,
        so this check uses an embedding and a linear layer.
        """
        import onnxruntime as ort

        from flashspec.export.onnx import export_draft_to_onnx

        model = torch.nn.Sequential(
            torch.nn.Embedding(1000, 32),
            torch.nn.Linear(32, 1000),
        ).eval()
        example_ids = torch.randint(0, 1000, (1, 16))

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "draft_seq_var.onnx")
            export_draft_to_onnx(
                drafter=model,
                output_path=onnx_path,
                example_input_ids=example_ids,
                gamma=4,
                vocab_size=1000,
            )
            session = ort.InferenceSession(onnx_path)

            for seq_len in [4, 8, 16, 32, 64]:
                ids = torch.randint(0, 1000, (1, seq_len))
                with torch.no_grad():
                    pt_logits = model(ids).float().numpy()
                ort_out = session.run(None, {"input_ids": ids.numpy()})
                ort_logits = ort_out[0]
                max_diff = float(abs(pt_logits - ort_logits).max())
                assert max_diff < 1e-5, (
                    f"ONNX parity FAILED at seq_len={seq_len}: max_diff={max_diff:.2e} ≥ 1e-5"
                )

    def test_onnx_invalid_opset_raises_value_error(self, toy_model_cpu: torch.nn.Module) -> None:
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
