"""Update committed benchmark results after a nightly run.

Called by `.github/workflows/benchmark.yml` after `make bench`.
Validates every JSON file in `benchmarks/results/` against the §14 schema,
then stages and commits the directory.

Usage
-----
    python scripts/update_benchmark_results.py
    python scripts/update_benchmark_results.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Required top-level keys in every result file (§14 schema).
REQUIRED_KEYS: frozenset[str] = frozenset({
    "schema_version",
    "timestamp",
    "git_commit",
    "hardware",
    "model",
    "method",
    "gamma",
    "dataset",
    "n_samples",
    "metrics",
})

# Required metric keys inside the "metrics" object.
REQUIRED_METRIC_KEYS: frozenset[str] = frozenset({
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
})

RESULTS_DIR: Path = Path("benchmarks/results")


def _validate_result_file(path: Path) -> list[str]:
    """Validate one result JSON file against the §14 schema.

    Parameters
    ----------
    path : Path
        Path to the JSON result file.

    Returns
    -------
    list[str]
        List of validation error messages (empty means valid).
    """
    errors: list[str] = []
    try:
        with open(path) as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        errors.append(f"  {path.name}: invalid JSON — {exc}")
        return errors

    missing_keys = REQUIRED_KEYS - set(data.keys())
    for key in sorted(missing_keys):
        errors.append(f"  {path.name}: missing top-level key '{key}'")

    if "metrics" in data and isinstance(data["metrics"], dict):
        missing_metrics = REQUIRED_METRIC_KEYS - set(data["metrics"].keys())
        for key in sorted(missing_metrics):
            errors.append(f"  {path.name}: missing metric key '{key}'")

    return errors


def _git_run(*cmd: str) -> str:
    """Run a git command and return stdout, raising on non-zero exit.

    Parameters
    ----------
    *cmd : str
        Git command arguments (without the leading ``'git'``).

    Returns
    -------
    str
        Decoded stdout of the command.

    Raises
    ------
    SystemExit
        If the git command returns a non-zero exit code.
    """
    result = subprocess.run(
        ["git", *cmd], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"git {' '.join(cmd)} failed:\n{result.stderr}\n"
        )
        sys.exit(1)
    return result.stdout.strip()


def main() -> None:
    """Validate result files, then commit and push benchmarks/results/."""
    parser = argparse.ArgumentParser(
        description="Validate and commit nightly benchmark results (AGENTS.md §12.3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate schema and print git commands without executing them.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(RESULTS_DIR),
        help=f"Results directory (default: {RESULTS_DIR}).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        sys.stdout.write(f"No JSON files found in {results_dir}.  Nothing to commit.\n")
        sys.exit(0)

    sys.stdout.write(f"Validating {len(json_files)} result file(s) ...\n")
    all_errors: list[str] = []
    for path in json_files:
        if path.name == "baseline.json":
            continue  # baseline is not a result file.
        errors = _validate_result_file(path)
        all_errors.extend(errors)

    if all_errors:
        sys.stderr.write("Schema validation FAILED:\n")
        for err in all_errors:
            sys.stderr.write(err + "\n")
        sys.exit(1)

    sys.stdout.write(f"All {len(json_files)} files are valid.\n")

    # Stage, commit, and push.
    git_date = subprocess.run(
        ["date", "+%Y-%m-%d"], capture_output=True, text=True
    ).stdout.strip()

    commit_msg = f"bench: nightly results {git_date}"

    if args.dry_run:
        sys.stdout.write("[DRY-RUN] Would run:\n")
        sys.stdout.write(f"  git add {results_dir}\n")
        sys.stdout.write(f"  git commit -m '{commit_msg}'\n")
        sys.stdout.write("  git push\n")
        sys.exit(0)

    _git_run("add", str(results_dir))

    # Only commit if there are staged changes.
    status = _git_run("status", "--porcelain", str(results_dir))
    if not status:
        sys.stdout.write("No changes to commit.\n")
        sys.exit(0)

    _git_run("commit", "-m", commit_msg)
    _git_run("push")
    sys.stdout.write(f"Committed and pushed: {commit_msg}\n")


if __name__ == "__main__":
    main()
