"""Width-7 diagnostic on the T4 target (Qwen2.5-1.5B), on bundle 3e38251's divergent prompts.

    pytest tests/integration/test_width_gpu.py -m gpu -v --no-cov

fp32 is the correctness check: greedy FlashSpec at verify width 7 (full and
4-of-7 padded), width 5 and unpadded must equal HuggingFace eager and
compiled greedy, token for token. Pad content must never reach a real row,
in fp32 and in fp16 (bit-identical row 0). The fp16 margins at each
divergence are written by ``benchmarks/width_diagnostic.py`` (notebook cell
3e); they are a report, not an assertion.
"""

from __future__ import annotations

import pytest

from benchmarks.width_diagnostic import diagnose

pytestmark = pytest.mark.gpu


def test_fp32_width_paths_are_token_exact_and_pads_never_leak() -> None:
    """fp32: every path matches both HuggingFace references; row 0 ignores the other slots."""
    report = diagnose(dtypes=("float32",))
    failures = [
        f"{run['prompt_id']} {name}: {path}"
        for run in report["cases"]
        for name, path in run["paths"].items()
        if path.get("divergence_vs_hf_static") is not None
        or path.get("divergence_vs_hf_eager") is not None
    ]
    assert not failures, "\n".join(failures)
    assert report["pad_independence_all_bit_identical"]


def test_fp16_pads_never_leak() -> None:
    """fp16: whatever the rounding, the other slots of a width-7 call never change row 0."""
    report = diagnose(dtypes=("float16",))
    assert report["pad_independence_all_bit_identical"], [
        run["pad_independence_at_bundle_divergence"] for run in report["cases"]
    ]
