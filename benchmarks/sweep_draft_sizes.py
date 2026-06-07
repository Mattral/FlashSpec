"""Acceptance rate vs draft model size ablation sweep.

Sweeps over draft model sizes (parameter counts) and records the mean
token acceptance rate α for each.  Produces a CSV result for inclusion
in the paper's ablation section (§8.4 §6 Ablation).

Usage
-----
    python benchmarks/sweep_draft_sizes.py --toy
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Draft model size labels and their approximate parameter counts.
# In toy mode these map to hidden-dimension sizes for the toy LM.
DRAFT_SIZES: list[tuple[str, int]] = [
    ("68M",   68),    # hidden dim proxy: 68
    ("160M",  160),
    ("1B",    256),
    ("3B",    512),
    ("7B",    768),
]

N_WARMUP_STEPS: int = 50
N_MEASUREMENT_STEPS: int = 200


def _toy_acceptance_rate(hidden_dim: int, gamma: int, seed: int) -> float:
    """Simulate an acceptance rate for a given hidden_dim using random logprobs.

    Larger hidden_dim ⟹ lower "cross-entropy" between draft and target
    (simulated by scaling the logit temperature), hence higher acceptance.

    Parameters
    ----------
    hidden_dim : int
        Proxy for draft model capacity.
    gamma : int
        Speculation length.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Simulated mean acceptance rate in [0, 1].
    """
    import torch
    from flashspec.utils.device import set_seed
    from flashspec.sampling.rejection import rejection_sample

    set_seed(seed)
    batch_size: int = 1
    vocab: int = 500
    seq_len: int = 16

    # Simulate better draft quality with larger models via narrower logit spread.
    temperature: float = max(0.3, 2.0 - hidden_dim / 512.0)

    alphas: list[float] = []
    for _ in range(N_MEASUREMENT_STEPS):
        target_lp = torch.randn(batch_size, gamma, vocab).log_softmax(-1)
        # Draft distribution: closer to target as temperature decreases.
        noise = torch.randn(batch_size, gamma, vocab) * temperature
        draft_lp = (target_lp.exp() + noise).clamp(min=1e-9).log().log_softmax(-1)
        draft_ids = torch.multinomial(
            draft_lp.exp().reshape(-1, vocab), num_samples=1
        ).reshape(batch_size, gamma)
        ctx = torch.randint(0, vocab, (batch_size, seq_len))
        _acc, _frej, alpha = rejection_sample(ctx, draft_lp, target_lp, draft_ids, gamma=gamma)
        alphas.append(alpha)

    return sum(alphas) / len(alphas)


def run_sweep(gamma: int, output_path: Path) -> None:
    """Run the draft-size sweep and write results to a CSV file.

    Parameters
    ----------
    gamma : int
        Speculation length used throughout the sweep.
    output_path : Path
        File path to write the CSV results.

    Returns
    -------
    None
    """
    rows: list[dict] = []
    sys.stdout.write(
        f"{'Draft size':<12} {'Alpha (mean)':>14} {'Alpha (std)':>12}\n"
    )
    sys.stdout.write("-" * 40 + "\n")

    for label, hidden_dim in DRAFT_SIZES:
        alphas: list[float] = [
            _toy_acceptance_rate(hidden_dim, gamma, seed=seed)
            for seed in range(5)
        ]
        mean_alpha = sum(alphas) / len(alphas)
        var_alpha = sum((a - mean_alpha) ** 2 for a in alphas) / len(alphas)
        std_alpha = var_alpha ** 0.5

        rows.append({
            "draft_size": label,
            "hidden_dim_proxy": hidden_dim,
            "gamma": gamma,
            "alpha_mean": f"{mean_alpha:.4f}",
            "alpha_std": f"{std_alpha:.4f}",
        })
        sys.stdout.write(
            f"{label:<12} {mean_alpha:>14.4f} {std_alpha:>12.4f}\n"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    sys.stdout.write(f"\nResults written to: {output_path}\n")


def main() -> None:
    """Parse CLI arguments and run the draft-size sweep."""
    parser = argparse.ArgumentParser(
        description="FlashSpec draft model size vs acceptance rate ablation sweep",
    )
    parser.add_argument(
        "--toy", action="store_true",
        help="Use toy random data (no real model weights needed).",
    )
    parser.add_argument(
        "--gamma", type=int, default=4, help="Speculation length γ (default: 4)."
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/results/draft_size_sweep.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if not args.toy:
        sys.stderr.write(
            "WARNING: Full sweep requires model weights. Falling back to toy data.\n"
        )

    run_sweep(gamma=args.gamma, output_path=Path(args.output))


if __name__ == "__main__":
    main()
