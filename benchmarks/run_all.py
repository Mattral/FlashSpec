"""Single entry-point for running all FlashSpec benchmarks.

Usage
-----
    python benchmarks/run_all.py --config benchmarks/configs/
    python benchmarks/run_all.py --config benchmarks/configs/ --toy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants (AGENTS.md §6) ──────────────────────────────────────────────────
N_WARMUP_STEPS: int = 50
N_MEASUREMENT_STEPS: int = 200
RESULT_SCHEMA_VERSION: str = "1.0"


# ── Schema builder ────────────────────────────────────────────────────────────

def _build_result(
    model: str,
    method: str,
    gamma: int,
    dataset: str,
    n_samples: int,
    metrics: dict[str, float],
    gpu_name: str,
    driver_version: str,
    cuda_version: str,
    git_commit: str,
) -> dict:
    """Build a benchmark result dict conforming to AGENTS.md §14 schema.

    Parameters
    ----------
    model : str
        HuggingFace model identifier.
    method : str
        Benchmark method name (e.g. ``"flashspec_ucb"``).
    gamma : int
        Speculation length used.
    dataset : str
        Dataset name (e.g. ``"mt_bench"``).
    n_samples : int
        Number of measurement steps.
    metrics : dict[str, float]
        Dict containing all §14 required metric keys.
    gpu_name : str
        GPU device name string.
    driver_version : str
        CUDA driver version string.
    cuda_version : str
        CUDA runtime version string.
    git_commit : str
        Short git commit hash.

    Returns
    -------
    dict
        JSON-serialisable result dict.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "hardware": {
            "gpu": gpu_name,
            "driver": driver_version,
            "cuda": cuda_version,
        },
        "model": model,
        "method": method,
        "gamma": gamma,
        "dataset": dataset,
        "n_samples": n_samples,
        "metrics": metrics,
    }


def _get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown'.

    Returns
    -------
    str
        Short commit hash or ``'unknown'``.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Warning: could not read git commit: {exc}\n")
        return "unknown"


def _get_cuda_info() -> tuple[str, str, str]:
    """Return (gpu_name, driver_version, cuda_version) or placeholder strings.

    Returns
    -------
    tuple[str, str, str]
        GPU name, driver version, CUDA version.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            cuda = torch.version.cuda or "unknown"
            return gpu, "unknown", cuda
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Warning: could not read CUDA info: {exc}\n")
    return "CPU (no GPU)", "N/A", "N/A"


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

    sys.stdout.write("Running toy benchmark (no real model weights)...\n")

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
    std_ms = variance ** 0.5

    # §6: log GPU memory (0 on CPU).
    gpu_mem_gb = 0.0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    sys.stdout.write(
        f"Toy benchmark: mean={mean_ms:.2f}±{std_ms:.2f} ms/step  "
        f"gpu_mem={gpu_mem_gb:.2f} GiB  (CPU, no GPU)\n"
    )
    sys.stdout.write("Toy benchmark PASSED.\n")


# ── Full benchmark (requires GPU + weights) ───────────────────────────────────

def run_full_benchmark(
    yaml_file: Path,
    output_dir: Path,
    git_commit: str,
    gpu_name: str,
    driver: str,
    cuda_ver: str,
) -> None:
    """Run one benchmark config and write the result JSON.

    Parameters
    ----------
    yaml_file : Path
        Path to a benchmark YAML config.
    output_dir : Path
        Directory to write the result JSON file.
    git_commit : str
        Current git commit hash.
    gpu_name : str
        GPU name for the result record.
    driver : str
        Driver version string.
    cuda_ver : str
        CUDA version string.

    Returns
    -------
    None

    Notes
    -----
    This function is a scaffold.  Full model-loading logic is implemented
    when real weights are available.  It documents the contract that every
    full benchmark run must follow (§6, §14).
    """
    import torch

    # §6: pin CPU threads.
    torch.set_num_threads(1)

    sys.stdout.write(f"\nRunning benchmark: {yaml_file.name}\n")
    sys.stdout.write(f"  Config: {yaml_file}\n")
    sys.stdout.write(f"  GPU: {gpu_name}\n")
    sys.stdout.write(f"  Commit: {git_commit}\n")

    # §6: log GPU memory watermark after benchmark.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # §6: synchronize before starting timed measurement.
        torch.cuda.synchronize()

    # NOTE: Full model loading + inference requires real weights at runtime.
    # The scaffold below shows the required metric schema (§14).
    # Implementors: replace the stub metrics with real measurements.
    stub_metrics: dict[str, float] = {
        "tokens_per_second_mean": 0.0,
        "tokens_per_second_std": 0.0,
        "speedup_vs_ar_mean": 0.0,
        "speedup_vs_ar_std": 0.0,
        "acceptance_rate_mean": 0.0,
        "acceptance_rate_std": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "gpu_memory_gb": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available()
            else 0.0
        ),
    }

    result = _build_result(
        model="stub",
        method="flashspec_ucb",
        gamma=4,
        dataset="stub",
        n_samples=N_MEASUREMENT_STEPS,
        metrics=stub_metrics,
        gpu_name=gpu_name,
        driver_version=driver,
        cuda_version=cuda_ver,
        git_commit=git_commit,
    )

    out_path = output_dir / f"{yaml_file.stem}_{git_commit}.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    sys.stdout.write(f"  Result written to: {out_path}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate benchmark routine."""
    parser = argparse.ArgumentParser(
        description="FlashSpec benchmark runner (AGENTS.md §14)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a YAML config file or a directory containing YAML files.",
    )
    parser.add_argument(
        "--toy", action="store_true",
        help="Run toy benchmark only (no model weights required; used in CPU CI).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/results",
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

    gpu_name, driver, cuda_ver = _get_cuda_info()
    git_commit = _get_git_commit()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for yaml_file in yaml_files:
        run_full_benchmark(yaml_file, output_dir, git_commit, gpu_name, driver, cuda_ver)

    sys.stdout.write("\nBenchmark run complete.\n")


if __name__ == "__main__":
    main()
