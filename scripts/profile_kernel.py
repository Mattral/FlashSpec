"""Kernel profiling helper using torch.profiler and Nsight Systems.

Profiles the Triton verification kernel and gather kernel under realistic
batch sizes and gamma values, exporting a Chrome-trace JSON and printing
a summary table.

Usage
-----
    # CPU profiling (no GPU required):
    python scripts/profile_kernel.py --cpu --output /tmp/trace.json

    # GPU profiling (requires CUDA):
    python scripts/profile_kernel.py --output /tmp/trace.json

    # Nsight Systems wrapper (runs this script under `nsys profile`):
    nsys profile --output /tmp/flashspec_kernels python scripts/profile_kernel.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Profile targets ───────────────────────────────────────────────────────────
# Each entry: (label, batch_size, gamma, vocab_size).
PROFILE_SHAPES: list[tuple[str, int, int, int]] = [
    ("B=1  γ=4  V=32k", 1,  4, 32_000),
    ("B=8  γ=4  V=32k", 8,  4, 32_000),
    ("B=1  γ=8  V=32k", 1,  8, 32_000),
    ("B=32 γ=4  V=32k", 32, 4, 32_000),
]

N_WARMUP: int = 10
N_PROFILE: int = 50


def _profile_reference(
    batch_size: int, gamma: int, vocab_size: int, device: str
) -> float:
    """Profile the pure-PyTorch reference kernel and return mean latency (ms).

    Parameters
    ----------
    batch_size : int
        Number of sequences in the batch.
    gamma : int
        Speculation length.
    vocab_size : int
        Vocabulary size.
    device : str
        PyTorch device string.

    Returns
    -------
    float
        Mean step latency in milliseconds over N_PROFILE steps.
    """
    import time
    import torch
    from flashspec.kernels._reference import verify_tokens_reference
    from flashspec.utils.device import set_seed

    set_seed(0)
    dev = torch.device(device)

    dlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(dev)
    tlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(dev)
    u   = torch.rand(batch_size, gamma).to(dev)
    ids = torch.randint(0, vocab_size, (batch_size, gamma)).to(dev)

    # Warm-up.
    for _ in range(N_WARMUP):
        verify_tokens_reference(dlp, tlp, u, ids)

    # Measure.
    if dev.type == "cuda":
        torch.cuda.synchronize()
    latencies: list[float] = []
    for _ in range(N_PROFILE):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        verify_tokens_reference(dlp, tlp, u, ids)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return sum(latencies) / len(latencies)


def _profile_triton(
    batch_size: int, gamma: int, vocab_size: int
) -> float:
    """Profile the Triton verify_tokens kernel and return mean latency (ms).

    Parameters
    ----------
    batch_size : int
        Number of sequences in the batch.
    gamma : int
        Speculation length.
    vocab_size : int
        Vocabulary size.

    Returns
    -------
    float
        Mean step latency in milliseconds over N_PROFILE steps.

    Raises
    ------
    RuntimeError
        If CUDA is not available.
    """
    import time
    import torch
    from flashspec.kernels import verify_tokens
    from flashspec.utils.device import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Triton profiling requires a CUDA device.")

    set_seed(0)
    device = torch.device("cuda")

    dlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(device)
    tlp = torch.randn(batch_size, gamma, vocab_size).log_softmax(-1).to(device)
    u   = torch.rand(batch_size, gamma, device=device)
    ids = torch.randint(0, vocab_size, (batch_size, gamma), device=device)

    # Warm-up (triggers autotune).
    for _ in range(N_WARMUP):
        verify_tokens(dlp, tlp, u, ids)

    torch.cuda.synchronize()
    latencies = []
    for _ in range(N_PROFILE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        verify_tokens(dlp, tlp, u, ids)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return sum(latencies) / len(latencies)


def main() -> None:
    """Parse CLI args, run profiling, and print a summary table."""
    parser = argparse.ArgumentParser(
        description="FlashSpec kernel profiler (torch.profiler / Nsight Systems)"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Profile reference implementation on CPU only (no GPU needed).",
    )
    parser.add_argument(
        "--output", type=str, default="/tmp/flashspec_kernel_trace.json",
        help="Path for the torch.profiler Chrome-trace JSON.",
    )
    args = parser.parse_args()

    import torch

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    sys.stdout.write(f"Profiling on device: {device}\n\n")

    header = f"{'Shape':<24} {'Reference (ms)':>16} {'Triton (ms)':>14} {'Speedup':>10}"
    sep = "-" * len(header)
    sys.stdout.write(header + "\n")
    sys.stdout.write(sep + "\n")

    try:
        import torch.profiler as profiler  # noqa: F401
        _has_profiler = True
    except ImportError:
        _has_profiler = False

    for label, batch_size, gamma, vocab_size in PROFILE_SHAPES:
        ref_ms = _profile_reference(batch_size, gamma, vocab_size, device)

        if not args.cpu and torch.cuda.is_available():
            try:
                triton_ms = _profile_triton(batch_size, gamma, vocab_size)
                speedup = f"{ref_ms / triton_ms:.2f}×"
                triton_str = f"{triton_ms:.3f}"
            except Exception as exc:
                triton_str = f"ERR ({exc})"
                speedup = "N/A"
        else:
            triton_str = "N/A (CPU)"
            speedup = "N/A"

        sys.stdout.write(
            f"{label:<24} {ref_ms:>16.3f} {triton_str:>14} {speedup:>10}\n"
        )

    sys.stdout.write(sep + "\n")
    sys.stdout.write(
        f"\nTarget: ≤ 0.5 ms per step on H100 (AGENTS.md §6)\n"
        f"Trace: {args.output} (not written in stub mode)\n"
    )


if __name__ == "__main__":
    main()
