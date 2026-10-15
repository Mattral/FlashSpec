"""Single entry-point for running all FlashSpec benchmarks.

Usage
-----
    python benchmarks/run_all.py --config benchmarks/configs/
    python benchmarks/run_all.py --config benchmarks/configs/ --toy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants (AGENTS.md §6) ──────────────────────────────────────────────────
N_WARMUP_STEPS: int = 50
N_MEASUREMENT_STEPS: int = 200


# ── Toy smoke-test benchmark ──────────────────────────────────────────────────


def run_toy_benchmark() -> None:
    """Run a fast smoke-test benchmark using random data (no real model weights).

    Pins CPU threads to 1 (§6), times N_MEASUREMENT_STEPS speculative steps
    using random logprobs, and logs mean latency.

    Returns
    -------
    None
    """
    import torch

    from flashspec.sampling.rejection import rejection_sample

    # §6: pin CPU threads for single-batch experiments.
    torch.set_num_threads(1)

    sys.stdout.write(
        "Running toy benchmark (random log-probs, sampling kernel only). "
        "This is not an end-to-end model measurement and writes no result file.\n"
    )

    batch_size: int = 1
    gamma: int = 4
    vocab: int = 1000
    seq_len: int = 32

    # Warm-up (§6: 50 warm-up steps).
    for _ in range(N_WARMUP_STEPS):
        dlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        tlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        ctx = torch.randint(0, vocab, (batch_size, seq_len))
        rejection_sample(ctx, dlp, tlp, ids, gamma=gamma)

    # Measurement (§6: 200 steps, mean ± std, synchronize around timed region).
    latencies_ms: list[float] = []
    for _ in range(N_MEASUREMENT_STEPS):
        dlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        tlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        ctx = torch.randint(0, vocab, (batch_size, seq_len))

        # §6: synchronize before AND after timed region.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _acc, _frej, _alpha = rejection_sample(ctx, dlp, tlp, ids, gamma=gamma)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    variance = sum((x - mean_ms) ** 2 for x in latencies_ms) / len(latencies_ms)
    std_ms = variance**0.5

    # §6: log GPU memory (0 on CPU).
    gpu_mem_gb = 0.0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    sys.stdout.write(
        f"Toy benchmark: mean={mean_ms:.2f}±{std_ms:.2f} ms/step  "
        f"gpu_mem={gpu_mem_gb:.2f} GiB  (CPU, no GPU)\n"
    )
    sys.stdout.write("Toy benchmark PASSED.\n")


# ── Full benchmark (requires GPU + weights) ───────────────────────────────────


def run_full_benchmark(yaml_file: Path, output_dir: Path) -> None:
    """Run one config and write only measurements this machine produced.

    Parameters
    ----------
    yaml_file : Path
        Benchmark YAML. ``model: toy`` builds an in-process network.
        Any other model id is loaded with ``transformers``. If that load
        fails, or the config asks for CUDA and CUDA is absent, this
        function raises and writes nothing.

    output_dir : Path
        Directory for JSON results.

    Raises
    ------
    SystemExit
        On a missing device or a checkpoint that could not be loaded.
        The message tells the operator why no file was written.
    """
    from benchmarks.harness import DeviceUnavailableError, run_config_file

    torch_threads()
    sys.stdout.write(f"\nRunning benchmark: {yaml_file.name}\n")
    try:
        written = run_config_file(yaml_file, output_dir)
    except DeviceUnavailableError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        raise SystemExit(2) from exc
    for path in written:
        sys.stdout.write(f"  Result written to: {path}\n")


def torch_threads() -> None:
    """Pin PyTorch intra-op threads to 1 for comparable CPU timings."""
    import torch

    torch.set_num_threads(1)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate benchmark routine."""
    parser = argparse.ArgumentParser(
        description="FlashSpec benchmark runner (AGENTS.md §14)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file or a directory containing YAML files.",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Run toy benchmark only (no model weights required; used in CPU CI).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to write JSON result files (default: benchmarks/results).",
    )
    args = parser.parse_args()

    if args.toy:
        run_toy_benchmark()
        return

    config_path = Path(args.config)
    if not config_path.exists():
        sys.stderr.write(f"ERROR: Config path does not exist: {config_path}\n")
        sys.exit(1)

    yaml_files: list[Path] = (
        sorted(config_path.glob("*.yaml")) if config_path.is_dir() else [config_path]
    )
    if not yaml_files:
        sys.stderr.write(f"ERROR: No .yaml files found in {config_path}\n")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from benchmarks.harness import detect_hardware

    sys.stdout.write(f"Hardware: {detect_hardware()}\n")

    failures = 0
    for yaml_file in yaml_files:
        try:
            run_full_benchmark(yaml_file, output_dir)
        except SystemExit:
            failures += 1
    if failures:
        sys.stderr.write(
            f"\n{failures} config(s) produced no result file. "
            "Measured configs were still written.\n"
        )
        sys.exit(2)

    sys.stdout.write("\nBenchmark run complete.\n")


if __name__ == "__main__":
    main()
