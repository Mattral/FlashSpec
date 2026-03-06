"""CLI: export a registered draft model to ONNX format.

Loads a draft model from the FlashSpec registry (or a local HuggingFace
checkpoint), exports it to ONNX, and verifies numerical parity against
the PyTorch baseline.

Usage
-----
    python scripts/export_draft.py \\
        --drafter llama3-1b \\
        --output draft.onnx \\
        --gamma 4 \\
        --batch-size 1

    python scripts/export_draft.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _verify_parity(
    onnx_path: str,
    pytorch_model: object,
    gamma: int,
    batch_sizes: list[int],
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX export parity against PyTorch for multiple batch sizes.

    Parameters
    ----------
    onnx_path : str
        Path to the exported ONNX file.
    pytorch_model : object
        PyTorch model with a ``forward(input_ids)`` method.
    gamma : int
        Speculation length (used to set sequence length).
    batch_sizes : list[int]
        Batch sizes to test.
    atol : float
        Absolute tolerance for logit comparison (default 1e-5, AGENTS.md §2.4).

    Returns
    -------
    bool
        ``True`` if parity holds for all batch sizes, ``False`` otherwise.
    """
    import torch
    try:
        import onnxruntime as ort  # type: ignore[import]
    except ImportError:
        sys.stderr.write(
            "onnxruntime not installed. Install with: pip install flashspec[onnx]\n"
        )
        return False

    session = ort.InferenceSession(onnx_path)
    all_ok = True

    for batch_size in batch_sizes:
        ids = torch.randint(0, 1000, (batch_size, gamma))
        with torch.no_grad():
            pt_out = pytorch_model(ids)  # type: ignore[operator]
            pt_logits = pt_out.logits.float().numpy()

        ort_out = session.run(None, {"input_ids": ids.numpy()})
        ort_logits = ort_out[0]

        max_diff = abs(pt_logits - ort_logits).max()
        status = "PASS" if max_diff < atol else "FAIL"
        sys.stdout.write(
            f"  batch_size={batch_size}: max_diff={max_diff:.2e} [{status}]\n"
        )
        if max_diff >= atol:
            all_ok = False

    return all_ok


def main() -> None:
    """Parse CLI arguments and run the ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export a FlashSpec draft model to ONNX (AGENTS.md §2.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--drafter", type=str, required=True,
        help="Drafter registry key (e.g. 'llama3-1b') or HuggingFace model id.",
    )
    parser.add_argument(
        "--output", type=str, default="draft.onnx",
        help="Output ONNX file path (default: draft.onnx).",
    )
    parser.add_argument(
        "--gamma", type=int, default=4,
        help="Speculation length γ used to set input sequence length (default: 4).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Example batch size for tracing (default: 1).",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32_000,
        help="Vocabulary size of the draft model (default: 32000).",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (minimum 17, AGENTS.md §2.4; default: 17).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify ONNX parity against PyTorch after export (requires onnxruntime).",
    )
    args = parser.parse_args()

    import torch
    from flashspec.export.onnx import export_draft_to_onnx

    sys.stdout.write(f"Exporting drafter '{args.drafter}' to {args.output} ...\n")

    # Attempt to load from registry first; fall back to HuggingFace if not found.
    try:
        from flashspec.engine.drafter import get_drafter
        drafter_cls = get_drafter(args.drafter)
        # Registry drafters are expected to accept no constructor args in toy mode.
        drafter_model: torch.nn.Module = drafter_cls()  # type: ignore[call-arg]
        sys.stdout.write(f"  Loaded from registry: {args.drafter}\n")
    except KeyError:
        sys.stderr.write(
            f"Drafter '{args.drafter}' not in registry.  "
            "Attempting HuggingFace load (requires network and model weights).\n"
        )
        try:
            from transformers import AutoModelForCausalLM  # type: ignore[import]
            drafter_model = AutoModelForCausalLM.from_pretrained(
                args.drafter, torch_dtype=torch.float32
            )
            sys.stdout.write(f"  Loaded from HuggingFace: {args.drafter}\n")
        except Exception as exc:
            sys.stderr.write(
                f"ERROR: Could not load drafter '{args.drafter}': {exc}\n"
                "Register it with @flashspec.engine.drafter.register() or "
                "provide a valid HuggingFace model id.\n"
            )
            sys.exit(1)

    example_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.gamma))

    try:
        export_draft_to_onnx(
            drafter=drafter_model,
            output_path=args.output,
            example_input_ids=example_ids,
            gamma=args.gamma,
            vocab_size=args.vocab_size,
            opset_version=args.opset,
        )
        sys.stdout.write(f"Export successful: {args.output}\n")
    except Exception as exc:
        sys.stderr.write(f"ERROR: ONNX export failed: {exc}\n")
        sys.exit(1)

    if args.verify:
        sys.stdout.write("\nVerifying ONNX parity (atol=1e-5, batch sizes 1..32):\n")
        batch_sizes = list(range(1, 33))
        ok = _verify_parity(
            onnx_path=args.output,
            pytorch_model=drafter_model,
            gamma=args.gamma,
            batch_sizes=batch_sizes,
        )
        if ok:
            sys.stdout.write("All parity checks PASSED.\n")
        else:
            sys.stderr.write("ONNX parity FAILED — check atol tolerance.\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
