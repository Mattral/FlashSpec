"""Check for performance regressions against the committed baseline.

Exit code 0 = no regression; exit code 1 = regression detected.

Usage
-----
    python scripts/check_regression.py --threshold 0.05
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _load_results(results_dir: Path) -> list[dict]:
    """Load all JSON result files from ``results_dir``."""
    results = []
    for path in sorted(glob.glob(str(results_dir / "*.json"))):
        with open(path) as f:
            results.append(json.load(f))
    return results


def check_regressions(results_dir: Path, threshold: float) -> bool:
    """Compare the latest result against the committed baseline.

    Parameters
    ----------
    results_dir : Path
        Directory containing JSON result files.
    threshold : float
        Fractional regression threshold (e.g. 0.05 = 5% slowdown allowed).

    Returns
    -------
    bool
        ``True`` if no regressions found, ``False`` otherwise.
    """
    baseline_path = results_dir / "baseline.json"
    if not baseline_path.exists():
        print("WARNING: No baseline.json found. Skipping regression check.")
        return True

    with open(baseline_path) as f:
        baseline = json.load(f)

    results = _load_results(results_dir)
    if not results:
        print("WARNING: No result files found. Skipping regression check.")
        return True

    latest = results[-1]
    regressions = []

    baseline_tps = baseline.get("metrics", {}).get("tokens_per_second", None)
    latest_tps   = latest.get("metrics", {}).get("tokens_per_second", None)

    if baseline_tps is not None and latest_tps is not None:
        regression_frac = (baseline_tps - latest_tps) / baseline_tps
        if regression_frac > threshold:
            regressions.append(
                f"tokens_per_second regressed by {regression_frac*100:.1f}% "
                f"(baseline={baseline_tps:.1f}, latest={latest_tps:.1f}, "
                f"threshold={threshold*100:.0f}%)"
            )

    if regressions:
        print("PERFORMANCE REGRESSIONS DETECTED:")
        for msg in regressions:
            print(f"  - {msg}")
        return False

    print(f"No regressions detected (threshold={threshold*100:.0f}%).")
    return True


def main() -> None:
    """Parse CLI arguments and run the regression check."""
    parser = argparse.ArgumentParser(description="FlashSpec performance regression checker")
    parser.add_argument(
        "--results-dir", type=str, default="benchmarks/results",
        help="Directory containing benchmark result JSON files.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Fractional regression threshold (default: 0.05 = 5%).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ok = check_regressions(results_dir, args.threshold)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
