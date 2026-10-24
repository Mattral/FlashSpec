"""Compare FlashSpec against Medusa, EAGLE, and vanilla AR baselines.

Produces a JSON result file per method that conforms to the result schema,
enabling the paper's results table to be populated from committed artifacts.

Usage
-----
    python benchmarks/compare_baselines.py --toy
    python benchmarks/compare_baselines.py --config benchmarks/configs/llama3_8b.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────
N_WARMUP_STEPS: int = 50
N_MEASUREMENT_STEPS: int = 200

# Named so the toy table can say why each one has no number in this mode.
_UNMEASURED_IN_TOY: tuple[tuple[str, str], ...] = (
    ("vanilla_ar", "needs a target forward pass; not timed here"),
    ("medusa", "not implemented"),
    ("eagle", "not implemented"),
    ("flashspec_thompson", "same sampling kernel; bandit policy is not timed here"),
)


def run_toy_comparison() -> None:
    """Run a fast toy comparison using random logprobs — no real weights needed.

    Measures mean step latency for the rejection-sampling path (flashspec_ucb)
    and prints a summary table.  Other methods are listed as not measured.

    Returns
    -------
    None
    """
    import torch

    from flashspec.sampling.rejection import rejection_sample

    torch.set_num_threads(1)

    batch_size: int = 1
    gamma: int = 4
    vocab: int = 1000
    seq_len: int = 32

    sys.stdout.write("=== FlashSpec Baseline Comparison (toy mode) ===\n\n")

    # Warm-up.
    for _ in range(N_WARMUP_STEPS):
        dlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        tlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        ctx = torch.randint(0, vocab, (batch_size, seq_len))
        rejection_sample(ctx, dlp, tlp, ids, gamma=gamma)

    # Measure FlashSpec UCB path (sampling only; no bandit overhead in toy).
    latencies_ms: list[float] = []
    accept_rates: list[float] = []
    for _ in range(N_MEASUREMENT_STEPS):
        dlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        tlp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        ids = torch.randint(0, vocab, (batch_size, gamma))
        ctx = torch.randint(0, vocab, (batch_size, seq_len))
        t0 = time.perf_counter()
        _acc, _frej, alpha = rejection_sample(ctx, dlp, tlp, ids, gamma=gamma)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        accept_rates.append(alpha)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    mean_alpha = sum(accept_rates) / len(accept_rates)

    header = f"{'Method':<22} {'Tokens/s (mean)':>16} {'α (mean)':>10} {'p50 ms':>8}"
    sep = "-" * len(header)
    sys.stdout.write(header + "\n")
    sys.stdout.write(sep + "\n")

    # FlashSpec: real measurement.
    tokens_per_second = (gamma * 1000.0) / mean_ms
    sys.stdout.write(
        f"{'flashspec_ucb':<22} {tokens_per_second:>16.1f} {mean_alpha:>10.3f} {mean_ms:>8.2f}\n"
    )

    for method, reason in _UNMEASURED_IN_TOY:
        sys.stdout.write(f"{method:<22} not measured ({reason})\n")

    sys.stdout.write(sep + "\n")
    sys.stdout.write(
        "NOTE: the flashspec_ucb row is sampling-kernel latency on random "
        "log-probs, not end-to-end tokens per second. Medusa and EAGLE are "
        "not implemented. This mode does not write a result file.\n"
    )


def main() -> None:
    """Parse CLI arguments and run the baseline comparison."""
    parser = argparse.ArgumentParser(
        description="FlashSpec vs Medusa vs EAGLE vs vanilla AR comparison",
    )
    parser.add_argument("--toy", action="store_true", help="Run with random data only.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a benchmark YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to write JSON result files.",
    )
    args = parser.parse_args()

    if args.toy:
        run_toy_comparison()
        sys.stdout.write(
            "\nToy mode times the sampling kernel on random log-probs. "
            "Medusa and EAGLE are not implemented, so they have no numbers. "
            "This mode does not write a result file.\n"
        )
        return

    if args.config is None:
        sys.stderr.write(
            "Pass --config, or --toy for a sampling-kernel smoke test. "
            "Refusing to write zero-filled baseline numbers.\n"
        )
        sys.exit(2)

    from benchmarks.harness import DeviceUnavailableError, run_config_file

    try:
        written = run_config_file(Path(args.config), Path(args.output_dir))
    except (DeviceUnavailableError, RuntimeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        sys.exit(2)
    for path in written:
        sys.stdout.write(f"Wrote {path}\n")


if __name__ == "__main__":
    main()
