"""ONNX export utilities for FlashSpec draft models.

Exports a draft model's ``generate_draft`` forward pass to ONNX format,
enabling deployment with ONNX Runtime.

The ONNX export must satisfy the parity contract (AGENTS.md §2.4):

    max(|onnx_logits - pytorch_logits|) < 1e-5   (all batch sizes 1..32)

Verified by ``tests/integration/test_onnx_parity.py``.
"""

from __future__ import annotations

import torch

from flashspec.utils.logging import get_logger

__all__ = ["export_draft_to_onnx"]

logger = get_logger(__name__)

# ONNX opset version — must be >= 17 for modern attention operators.
_ONNX_OPSET_VERSION: int = 17


def export_draft_to_onnx(
    drafter: torch.nn.Module,
    output_path: str,
    example_input_ids: torch.Tensor,
    gamma: int,
    vocab_size: int,
    opset_version: int = _ONNX_OPSET_VERSION,
) -> None:
    """Export a draft model to ONNX format.

    Parameters
    ----------
    drafter : torch.nn.Module
        Draft model with a ``forward(input_ids)`` method returning logits of
        shape ``(batch_size, seq_len, vocab_size)``.
    output_path : str
        File path to write the ``.onnx`` file to.
    example_input_ids : torch.Tensor
        Example input tensor used for tracing.
        Shape: ``(batch_size, seq_len)``, dtype int64.
    gamma : int
        Speculation length (number of tokens to export as dynamic axis).
    vocab_size : int
        Vocabulary size of the draft model.
    opset_version : int
        ONNX opset version.  Minimum 17.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``opset_version`` < 17 or ``gamma`` < 1.
    RuntimeError
        If ONNX export fails for any reason.

    Notes
    -----
    Dynamic axes are set for ``batch_size`` and ``sequence_length`` to support
    variable-length inference at runtime.  The exported model is verified for
    numerical parity by ``tests/integration/test_onnx_parity.py``.

    Examples
    --------
    >>> export_draft_to_onnx(
    ...     drafter=my_draft_model,
    ...     output_path="draft.onnx",
    ...     example_input_ids=torch.randint(0, 1000, (1, 32)),
    ...     gamma=4,
    ...     vocab_size=32000,
    ... )
    """
    if opset_version < 17:
        raise ValueError(
            f"opset_version must be >= 17; got {opset_version}."
        )
    if gamma < 1:
        raise ValueError(f"gamma must be >= 1; got {gamma}.")

    drafter.eval()
    try:
        import onnx  # type: ignore[import]
        torch.onnx.export(
            drafter,
            (example_input_ids,),
            output_path,
            opset_version=opset_version,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            do_constant_folding=True,
        )
        model_proto = onnx.load(output_path)
        onnx.checker.check_model(model_proto)
        logger.debug("ONNX export successful", extra={"path": output_path})
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires 'onnx' and 'onnxruntime-gpu'. "
            "Install with: pip install flashspec[onnx]"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"ONNX export failed: {exc}") from exc
