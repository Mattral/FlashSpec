"""Time the CPU residual distribution and the CPU verification step.

Compares pure PyTorch with the optional C baseline and AVX2 kernel from
``flashspec-asm-kernel`` when that package's shared library actually
loads. Backends that are not available are omitted from the JSON.
Their absence is recorded under ``unmeasured`` with a reason. This
script does not copy the upstream Linux timings into the result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.harness import detect_hardware, git_commit  # noqa: E402
from flashspec.sampling.rejection import (  # noqa: E402
    rejection_sample,
    residual_backend_status,
    residual_distribution,
)

# End-to-end generation fields. This script times one CPU op and does not
# measure them. Listing them keeps the result schema valid.
_E2E_UNMEASURED: tuple[str, ...] = (
    "tokens_per_second_mean",
    "tokens_per_second_std",
    "speedup_vs_ar_mean",
    "speedup_vs_ar_std",
    "acceptance_rate_mean",
    "acceptance_rate_std",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "gpu_memory_gb",
)


def _time_call(fn: Callable[[], object], iterations: int, warmup: int) -> dict[str, float]:
    """Warm up ``fn`` and return mean and std nanoseconds... milliseconds.

    Parameters
    ----------
    fn : callable
        Zero-argument function to time.
    iterations : int
        Timed repetitions.
    warmup : int
        Untimed repetitions.

    Returns
    -------
    dict[str, float]
        ``mean_ms`` and ``std_ms``.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    mean = sum(samples) / len(samples)
    var = sum((item - mean) ** 2 for item in samples) / len(samples)
    return {"mean_ms": mean, "std_ms": var**0.5, "iterations": float(iterations)}


def _pytorch_residual_ms(vocab: int, iterations: int, warmup: int) -> dict[str, float]:
    """Time :func:`residual_distribution` with the assembly backend forced off.

    Parameters
    ----------
    vocab : int
        Vocabulary length of the slice.
    iterations : int
        Timed repetitions.
    warmup : int
        Untimed repetitions.

    Returns
    -------
    dict[str, float]
        Timing summary.
    """
    import torch

    import flashspec.sampling.rejection as rejection

    torch.manual_seed(0)
    p = torch.softmax(torch.randn(vocab), dim=-1)
    q = torch.softmax(torch.randn(vocab), dim=-1)
    previous = rejection._ASM_AVAILABLE
    rejection._ASM_AVAILABLE = False
    try:
        return _time_call(lambda: residual_distribution(p, q), iterations, warmup)
    finally:
        rejection._ASM_AVAILABLE = previous


def _kernel_ms(which: str, vocab: int, iterations: int, warmup: int) -> dict[str, float] | None:
    """Time one flashspec-asm entry point, or return ``None`` if it is absent.

    Parameters
    ----------
    which : str
        ``"asm"`` or ``"c_baseline"``.
    vocab : int
        Vocabulary length.
    iterations : int
        Timed repetitions.
    warmup : int
        Untimed repetitions.

    Returns
    -------
    dict[str, float] or None
        Timing summary, or ``None`` when the shared library is not loaded.
    """
    try:
        import flashspec_asm_kernel as kernel
        import numpy as np
    except (ImportError, OSError):
        return None
    try:
        if which == "asm":
            fn = kernel.residual_dist
            if not kernel.cpu_supports_avx2():
                return None
        else:
            fn = kernel.residual_dist_c_baseline
    except (AttributeError, ImportError, OSError, RuntimeError):
        return None
    rng = np.random.default_rng(0)
    p = rng.random(vocab, dtype=np.float32)
    q = rng.random(vocab, dtype=np.float32)
    p = p / p.sum()
    q = q / q.sum()
    return _time_call(lambda: fn(p, q), iterations, warmup)


def _verification_ms(vocab: int, iterations: int, warmup: int, use_asm: bool) -> dict[str, float]:
    """Time :func:`rejection_sample` on CPU log-probabilities.

    Parameters
    ----------
    vocab : int
        Vocabulary size.
    iterations : int
        Timed repetitions.
    warmup : int
        Untimed repetitions.
    use_asm : bool
        When ``False``, force the pure-PyTorch residual. When ``True``,
        leave the import-time selection in place. If the kernel was not
        selected, the caller should not treat the result as an ASM timing.

    Returns
    -------
    dict[str, float]
        Timing summary.
    """
    import torch

    import flashspec.sampling.rejection as rejection

    torch.manual_seed(0)
    batch, gamma = 1, 4
    draft = torch.randn(batch, gamma, vocab).log_softmax(-1)
    target = torch.randn(batch, gamma, vocab).log_softmax(-1)
    token_ids = torch.randint(0, vocab, (batch, gamma))
    context = torch.randint(0, vocab, (batch, 8))
    previous = rejection._ASM_AVAILABLE
    if not use_asm:
        rejection._ASM_AVAILABLE = False
    try:
        return _time_call(
            lambda: rejection_sample(context, draft, target, token_ids, gamma=gamma),
            iterations,
            warmup,
        )
    finally:
        rejection._ASM_AVAILABLE = previous


def run(vocabs: list[int], iterations: int, warmup: int) -> dict[str, object]:
    """Measure the requested vocabulary sizes.

    Parameters
    ----------
    vocabs : list[int]
        Vocabulary sizes to time.
    iterations : int
        Timed repetitions per cell.
    warmup : int
        Untimed repetitions per cell.

    Returns
    -------
    dict
        Result document. ASM and C cells are omitted when the library
        did not load.
    """
    import torch

    torch.set_num_threads(1)
    rows: list[dict[str, object]] = []
    unmeasured: list[str] = []
    for vocab in vocabs:
        row: dict[str, object] = {
            "vocab": vocab,
            "pytorch_residual": _pytorch_residual_ms(vocab, iterations, warmup),
            "pytorch_verification": _verification_ms(vocab, iterations, warmup, use_asm=False),
        }
        asm = _kernel_ms("asm", vocab, iterations, warmup)
        c_baseline = _kernel_ms("c_baseline", vocab, iterations, warmup)
        if asm is None:
            unmeasured.append(f"asm_residual_v{vocab}")
        else:
            row["asm_residual"] = asm
        if c_baseline is None:
            unmeasured.append(f"c_baseline_residual_v{vocab}")
        else:
            row["c_baseline_residual"] = c_baseline
        status = residual_backend_status()
        if status.get("backend") == "asm":
            row["selected_backend_verification"] = _verification_ms(
                vocab, iterations, warmup, use_asm=True
            )
        else:
            unmeasured.append(f"asm_verification_v{vocab}")
        rows.append(row)
    unmeasured.extend(_E2E_UNMEASURED)
    reason = str(residual_backend_status().get("reason"))
    return {
        "status": "measured",
        "measurement_kind": "cpu_residual_and_verification_microbench",
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "hardware": detect_hardware(),
        "model": "none",
        "method": "residual_microbench",
        "gamma": 4,
        "dataset": "random_logprobs",
        "n_samples": iterations,
        "metrics": {"rows": rows},
        "unmeasured": unmeasured,
        "residual_backend": residual_backend_status(),
        "notes": (
            "Pure-PyTorch numbers are measured on this machine. "
            "ASM and C-baseline numbers are present only when "
            "flashspec-asm-kernel loaded its shared library. "
            f"Backend reason: {reason}. "
            "Upstream Linux timings published in flashspec-asm are not "
            "copied into this file. timestamp is the UTC time when this "
            "script finished the timings. Timings are Python-level "
            "perf_counter calls."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] or None
        Argument vector excluding the program name.
    """
    parser = argparse.ArgumentParser(description="CPU residual microbenchmark")
    parser.add_argument(
        "--output",
        default="benchmarks/results/residual_cpu.json",
        help="JSON destination.",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--vocab",
        type=int,
        nargs="+",
        default=[1024, 4096],
        help="Vocabulary sizes. Add 32000 for a longer run.",
    )
    args = parser.parse_args(argv)
    document = run(list(args.vocab), args.iterations, args.warmup)
    destination = Path(args.output)
    # write_result expects the harness document shape; this microbench
    # stores a list of rows under metrics, so dump it directly.
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")
    print(f"Backend: {document['residual_backend']}")


if __name__ == "__main__":
    main()
