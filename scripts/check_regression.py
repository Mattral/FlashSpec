"""Check for performance regressions against the committed baseline.

Implements the exact per-model, per-metric regression thresholds from
AGENTS.md §6.  Exits with code 0 if no regressions; code 1 if any
threshold is violated.

Usage
-----
    python scripts/check_regression.py
    python scripts/check_regression.py --results-dir benchmarks/results
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# ── Per-model regression thresholds (AGENTS.md §6) ───────────────────────────
# Key: substring that must appear in the result's "model" field.
# Value: dict of metric → minimum acceptable value (or maximum for latency).
# A result file is matched to the first key whose substring appears in model name.
_SPEEDUP_FLOORS: dict[str, float] = {
    "Llama-3-8B":  1.8,   # < 1.8× fails
    "Llama-3-70B": 2.2,   # < 2.2× fails
    "Mistral-7B":  1.7,   # < 1.7× fails
}

# Kernel and bandit overhead ceilings are stored as separate result methods.
_KERNEL_LATENCY_CEILING_MS: float = 1.0     # > 1 ms fails (§6)
_BANDIT_LATENCY_CEILING_US: float = 200.0   # > 200 µs fails (§6)
_GPU_MEMORY_OVERHEAD_CEILING: float = 0.25  # > 25% overhead fails (§6)


def _load_results(results_dir: Path) -> list[dict]:
    """Load all non-baseline JSON result files from ``results_dir``.

    Parameters
    ----------
    results_dir : Path
        Directory containing JSON benchmark result files.

    Returns
    -------
    list[dict]
        Parsed result dicts, sorted by timestamp ascending.
    """
    results = []
    for path in sorted(glob.glob(str(results_dir / "*.json"))):
        if Path(path).name == "baseline.json":
            continue
        try:
            with open(path) as fh:
                results.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            sys.stderr.write(f"Warning: could not read {path}: {exc}\n")
    return results


def _match_model(model_field: str) -> str | None:
    """Return the matching threshold key for a model field string, or None.

    Parameters
    ----------
    model_field : str
        The ``"model"`` field from a result JSON file.

    Returns
    -------
    str or None
        The first matching key from ``_SPEEDUP_FLOORS``, or ``None``.
    """
    for key in _SPEEDUP_FLOORS:
        if key in model_field:
            return key
    return None


def check_regressions(results_dir: Path) -> bool:
    """Compare the latest results against §6 thresholds.

    Checks:
    - Per-model ``speedup_vs_ar_mean`` against §6 regression floors.
    - ``p99_latency_ms`` for the kernel overhead ceiling (method == "kernel_profile").
    - ``gpu_memory_gb`` overhead ceiling.

    Parameters
    ----------
    results_dir : Path
        Directory containing benchmark result JSON files.

    Returns
    -------
    bool
        ``True`` if no regressions found, ``False`` if any threshold violated.
    """
    baseline_path = results_dir / "baseline.json"
    if not baseline_path.exists():
        sys.stderr.write(
            "Warning: baseline.json not found — cannot compute relative regressions.\n"
            "Checking absolute §6 thresholds only.\n"
        )

    results = _load_results(results_dir)
    if not results:
        sys.stdout.write(
            f"No result files found in {results_dir}.  Nothing to check.\n"
        )
        return True

    regressions: list[str] = []

    for result in results:
        model = result.get("model", "")
        method = result.get("method", "")
        metrics = result.get("metrics", {})

        # ── Speedup regression (per-model floor from §6) ──────────────────
        threshold_key = _match_model(model)
        if threshold_key is not None and method not in ("vanilla_ar",):
            speedup = metrics.get("speedup_vs_ar_mean", None)
            if speedup is not None and speedup > 0.0:
                floor = _SPEEDUP_FLOORS[threshold_key]
                if speedup < floor:
                    regressions.append(
                        f"[{model} / {method}] speedup_vs_ar_mean={speedup:.3f} "
                        f"< regression floor {floor:.1f}× (§6)"
                    )

        # ── Kernel latency ceiling (§6: ≤ 0.5 ms target, > 1 ms fails) ───
        if method == "kernel_profile":
            p99_ms = metrics.get("p99_latency_ms", None)
            if p99_ms is not None and p99_ms > _KERNEL_LATENCY_CEILING_MS:
                regressions.append(
                    f"[{model} / kernel_profile] p99_latency_ms={p99_ms:.2f} "
                    f"> ceiling {_KERNEL_LATENCY_CEILING_MS} ms (§6)"
                )

        # ── GPU memory overhead ceiling (§6: > 25% overhead fails) ────────
        # Compare against the baseline model's gpu_memory_gb if available.
        if baseline_path.exists():
            try:
                with open(baseline_path) as fh:
                    baseline = json.load(fh)
                baseline_mem = baseline.get("metrics", {}).get("gpu_memory_gb", 0.0)
                current_mem = metrics.get("gpu_memory_gb", 0.0)
                if baseline_mem > 0.0 and current_mem > 0.0:
                    overhead = (current_mem - baseline_mem) / baseline_mem
                    if overhead > _GPU_MEMORY_OVERHEAD_CEILING:
                        regressions.append(
                            f"[{model} / {method}] GPU memory overhead="
                            f"{overhead*100:.1f}% > {_GPU_MEMORY_OVERHEAD_CEILING*100:.0f}% (§6)"
                        )
            except (json.JSONDecodeError, OSError):
                pass  # baseline unreadable — skip relative memory check

    if regressions:
        sys.stderr.write(
            f"REGRESSION FAILURES ({len(regressions)}):\n"
        )
        for msg in regressions:
            sys.stderr.write(f"  ✗ {msg}\n")
        return False

    sys.stdout.write(
        f"No regressions detected against §6 thresholds "
        f"({len(results)} result file(s) checked).\n"
    )
    return True


def main() -> None:
    """Parse CLI arguments and run the regression check."""
    parser = argparse.ArgumentParser(
        description="FlashSpec performance regression checker (AGENTS.md §6)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="benchmarks/results",
        help=f"Directory containing benchmark JSON files (default: benchmarks/results).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.stderr.write(
            f"ERROR: results dir does not exist: {results_dir}\n"
        )
        sys.exit(1)

    ok = check_regressions(results_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
