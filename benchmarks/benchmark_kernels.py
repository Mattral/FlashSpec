"""Kernel benchmarks with roofline analysis for FlashSpec Triton kernels.

This module satisfies §13.5 requirement:
  "Kernel is benchmarked in benchmarks/ with roofline analysis."

The roofline model identifies whether each kernel is memory-bandwidth bound
or compute bound on the current GPU.  See docs/kernels.md for analysis.

Usage
-----
    python benchmarks/benchmark_kernels.py
    python benchmarks/benchmark_kernels.py --toy   # CPU, no GPU needed
    python benchmarks/benchmark_kernels.py --output benchmarks/results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Benchmark shapes (§13.5: representative of production use) ─────────────
# Each entry: (label, batch_size, gamma, vocab_size)
KERNEL_SHAPES: list[tuple[str, int, int, int]] = [
    ("B=1  γ=4  V=32k",  1,  4, 32_000),
    ("B=1  γ=8  V=32k",  1,  8, 32_000),
    ("B=8  γ=4  V=32k",  8,  4, 32_000),
    ("B=32 γ=4  V=32k", 32,  4, 32_000),
    ("B=1  γ=4  V=32768", 1,  4, 32_768),  # Mistral vocab
]

N_WARMUP: int = 20
N_STEPS: int = 200

# H100 SXM5 hardware roofline constants.
# Source: NVIDIA H100 Tensor Core GPU Architecture whitepaper.
_H100_PEAK_BANDWIDTH_TB_S: float = 3.35   # HBM3 bandwidth (TB/s)
_H100_PEAK_TFLOPS_BF16: float = 989.0     # TFLOP/s BF16 tensor core


def _bytes_for_verify_kernel(
    batch_size: int, gamma: int, vocab_size: int
) -> int:
    """Compute memory bytes accessed by one verify_tokens kernel call.

    The kernel reads two scalars per (batch, gamma) position:
    lp_q[b, i, token_id] and lp_p[b, i, token_id].
    It also reads token_ids and writes accepted + first_rejection.

    Parameters
    ----------
    batch_size : int
        Batch dimension.
    gamma : int
        Speculation length.
    vocab_size : int
        Vocabulary size (only two scalars per position are read, not the
        full vocab slice).

    Returns
    -------
    int
        Estimated bytes accessed per kernel call.

    Notes
    -----
    Memory-bandwidth roofline: bytes_accessed / peak_bandwidth = roofline_time.
    If measured_time > roofline_time, kernel is memory-bound.
    If measured_time ≈ roofline_time, kernel is at the memory bandwidth limit.
    """
    n = batch_size * gamma
    # 2 floats read (lp_q, lp_p) + 1 int64 (token_id) + 1 float (u)
    bytes_read = n * (4 + 4 + 8 + 4)
    # 1 bool (accepted) + 1 int32 (first_rejection)
    bytes_write = n * 1 + batch_size * 4
    return bytes_read + bytes_write


def _roofline_time_ms(
    bytes_accessed: int,
    peak_bandwidth_tb_s: float,
) -> float:
    """Compute the memory-bandwidth roofline lower-bound time in ms.

    Parameters
    ----------
    bytes_accessed : int
        Estimated bytes read + written by the kernel.
    peak_bandwidth_tb_s : float
        Peak HBM bandwidth in terabytes per second.

    Returns
    -------
    float
        Minimum achievable time in milliseconds (memory-bandwidth bound).
    """
    return (bytes_accessed / (peak_bandwidth_tb_s * 1e12)) * 1000.0


def _time_reference(
    batch_size: int, gamma: int, vocab_size: int, device_str: str
) -> tuple[float, float]:
    """Time the pure-PyTorch reference kernel and return (mean_ms, std_ms).

    Parameters
    ----------
    batch_size : int
        Batch dimension.
    gamma : int
        Speculation length.
    vocab_size : int
        Vocabulary size.
    device_str : str
        PyTorch device string (e.g. ``"cpu"`` or ``"cuda:0"``).

    Returns
    -------
    tuple[float, float]
        ``(mean_ms, std_ms)`` over ``N_STEPS`` steps after ``N_WARMUP`` warmup.
    """
    import torch
    from flashspec.kernels._reference import verify_tokens_reference
    from flashspec.utils.device import set_seed

    set_seed(0)
    device = torch.device(device_str)
    dlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(device)
    tlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(device)
    u   = torch.rand(batch_size, gamma, device=device)
    ids = torch.randint(0, vocab_size, (batch_size, gamma), device=device)

    for _ in range(N_WARMUP):
        verify_tokens_reference(dlp.cpu(), tlp.cpu(), u.cpu(), ids.cpu())

    lats: list[float] = []
    for _ in range(N_STEPS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        verify_tokens_reference(dlp.cpu(), tlp.cpu(), u.cpu(), ids.cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(lats) / len(lats)
    var = sum((x - mean_ms) ** 2 for x in lats) / len(lats)
    return mean_ms, var ** 0.5


def _time_triton(
    batch_size: int, gamma: int, vocab_size: int
) -> tuple[float, float]:
    """Time the Triton verify_tokens kernel and return (mean_ms, std_ms).

    Parameters
    ----------
    batch_size : int
        Batch dimension.
    gamma : int
        Speculation length.
    vocab_size : int
        Vocabulary size.

    Returns
    -------
    tuple[float, float]
        ``(mean_ms, std_ms)`` over ``N_STEPS`` measurement steps.

    Raises
    ------
    RuntimeError
        If CUDA is not available.
    """
    import torch
    from flashspec.kernels import verify_tokens
    from flashspec.utils.device import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Triton kernel benchmarking requires a CUDA device.")

    set_seed(0)
    device = torch.device("cuda:0")
    dlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(device)
    tlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(device)
    u   = torch.rand(batch_size, gamma, device=device)
    ids = torch.randint(0, vocab_size, (batch_size, gamma), device=device)

    # Warmup triggers autotune.
    for _ in range(N_WARMUP):
        verify_tokens(dlp, tlp, u, ids)

    torch.cuda.synchronize()
    lats: list[float] = []
    for _ in range(N_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        verify_tokens(dlp, tlp, u, ids)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(lats) / len(lats)
    var = sum((x - mean_ms) ** 2 for x in lats) / len(lats)
    return mean_ms, var ** 0.5


def run_roofline_benchmark(
    toy: bool,
    output_dir: Path | None,
) -> list[dict]:
    """Run the kernel roofline benchmark and return results as a list of dicts.

    Parameters
    ----------
    toy : bool
        If ``True``, run CPU-only reference timing (no GPU needed).
    output_dir : Path or None
        Directory to write per-shape JSON result files.  If ``None``,
        results are printed only.

    Returns
    -------
    list[dict]
        One dict per shape containing timing and roofline metrics.
    """
    import torch

    device_str = "cpu" if toy else ("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available() and not toy
        else "CPU"
    )

    sys.stdout.write(
        f"\nFlashSpec Kernel Roofline Benchmark\n"
        f"Device : {gpu_name}\n"
        f"Shapes : {len(KERNEL_SHAPES)}\n"
        f"{'Shape':<28} {'Ref (ms)':>10} {'Ref std':>8} "
        f"{'Triton (ms)':>12} {'Triton std':>10} "
        f"{'Roofline (ms)':>14} {'Ratio':>8}\n"
    )
    sys.stdout.write("-" * 96 + "\n")

    results: list[dict] = []
    for label, batch_size, gamma, vocab_size in KERNEL_SHAPES:
        ref_mean, ref_std = _time_reference(
            batch_size, gamma, vocab_size, device_str
        )
        bytes_acc = _bytes_for_verify_kernel(batch_size, gamma, vocab_size)
        roofline_ms = _roofline_time_ms(bytes_acc, _H100_PEAK_BANDWIDTH_TB_S)

        triton_mean = triton_std = 0.0
        if not toy and torch.cuda.is_available():
            try:
                triton_mean, triton_std = _time_triton(batch_size, gamma, vocab_size)
            except RuntimeError as exc:
                sys.stderr.write(f"  Triton unavailable: {exc}\n")

        # Efficiency ratio: roofline / measured (1.0 = at bandwidth limit)
        ratio = roofline_ms / triton_mean if triton_mean > 0 else 0.0

        sys.stdout.write(
            f"{label:<28} {ref_mean:>10.3f} {ref_std:>8.3f} "
            f"{triton_mean:>12.3f} {triton_std:>10.3f} "
            f"{roofline_ms:>14.4f} {ratio:>8.3f}\n"
        )

        row: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": {"gpu": gpu_name},
            "shape": {
                "batch_size": batch_size, "gamma": gamma, "vocab_size": vocab_size
            },
            "label": label,
            "reference_mean_ms": ref_mean,
            "reference_std_ms": ref_std,
            "triton_mean_ms": triton_mean,
            "triton_std_ms": triton_std,
            "roofline_lower_bound_ms": roofline_ms,
            "bytes_accessed": bytes_acc,
            "memory_efficiency_ratio": ratio,
            "n_warmup": N_WARMUP,
            "n_steps": N_STEPS,
        }
        results.append(row)

        if output_dir is not None:
            safe_label = label.replace(" ", "_").replace("=", "").replace("/", "")
            out_path = output_dir / f"kernel_bench_{safe_label}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(row, fh, indent=2)

    sys.stdout.write("-" * 96 + "\n")
    sys.stdout.write(
        "Ratio = roofline / triton.  Target ≥ 0.5 (§6: ≤ 0.5 ms per step on H100).\n"
        "Ratio close to 1.0 means kernel is operating at the HBM bandwidth limit.\n"
    )
    return results


def main() -> None:
    """Parse CLI arguments and run the kernel roofline benchmark."""
    parser = argparse.ArgumentParser(
        description="FlashSpec kernel roofline benchmark (§13.5)",
    )
    parser.add_argument(
        "--toy", action="store_true",
        help="CPU-only reference timing — no GPU or Triton needed.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to write JSON result files (default: print only).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None
    run_roofline_benchmark(toy=args.toy, output_dir=output_dir)


if __name__ == "__main__":
    main()
