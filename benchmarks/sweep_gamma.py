"""Gamma hyperparameter sweep for FlashSpec.

Sweeps gamma in [1, 2, 4, 8, 16] and records acceptance rate and
tokens/s for each value.  Produces a CSV result in benchmarks/results/.

Usage
-----
    python benchmarks/sweep_gamma.py --toy
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

_GAMMA_VALUES = [1, 2, 4, 8, 16]
_N_STEPS = 50
_VOCAB = 1000
_BATCH = 1
_SEQ_LEN = 32


def _toy_step(gamma: int) -> tuple[float, float]:
    """Run one toy speculative step and return (alpha, step_ms)."""
    from flashspec.sampling.rejection import rejection_sample

    dlp = torch.randn(_BATCH, gamma, _VOCAB).log_softmax(-1)
    tlp = torch.randn(_BATCH, gamma, _VOCAB).log_softmax(-1)
    ids = torch.randint(0, _VOCAB, (_BATCH, gamma))
    ctx = torch.randint(0, _VOCAB, (_BATCH, _SEQ_LEN))

    t0 = time.perf_counter()
    _acc, _frej, alpha = rejection_sample(ctx, dlp, tlp, ids, gamma=gamma)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return alpha, elapsed_ms


def run_sweep(output_path: Path) -> None:
    """Run the gamma sweep and write results to a CSV file."""
    rows: list[dict] = []
    for gamma in _GAMMA_VALUES:
        alphas, latencies = [], []
        for _ in range(_N_STEPS):
            a, ms = _toy_step(gamma)
            alphas.append(a)
            latencies.append(ms)

        mean_alpha = sum(alphas) / len(alphas)
        mean_ms = sum(latencies) / len(latencies)
        mean_tps = (gamma * 1000.0) / mean_ms if mean_ms > 0 else 0.0

        rows.append({
            "gamma": gamma,
            "mean_acceptance_rate": f"{mean_alpha:.4f}",
            "mean_step_ms": f"{mean_ms:.3f}",
            "mean_tokens_per_second": f"{mean_tps:.1f}",
        })
        print(f"gamma={gamma:2d}  α={mean_alpha:.3f}  {mean_ms:.2f}ms  {mean_tps:.0f}t/s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to: {output_path}")


def main() -> None:
    """CLI entry point for the gamma sweep."""
    parser = argparse.ArgumentParser(description="FlashSpec gamma sweep")
    parser.add_argument("--toy", action="store_true", help="Use toy random data.")
    parser.add_argument(
        "--output", type=str, default="benchmarks/results/gamma_sweep.csv",
        help="Output CSV path."
    )
    args = parser.parse_args()

    if not args.toy:
        print("WARNING: Full sweep requires model weights. Using --toy data.", file=sys.stderr)

    run_sweep(Path(args.output))


if __name__ == "__main__":
    main()
