r"""Run benchmark configs and pack every result into one downloadable zip.

Usage (Colab, from the checkout root)::

    python benchmarks/gpu_bundle.py \
        --config benchmarks/configs/qwen_t4.yaml \
        --config benchmarks/configs/tinyllama_t4.yaml \
        --output-dir /content/fs_bundle

The zip ``flashspec_gpu_bundle_<commit>.zip`` (next to ``--output-dir``)
holds:

- ``results/*.json``: the harness result files, unedited.
- ``manifest.json``: commit, whether tracked files were modified, per-config
  status and duration, versions, hardware (the harness fields only), and
  a sha256 for every file.
- ``profile.txt``: a plain-text view of the step profiles, forward counts,
  host reads and tok/s, printed from the JSONs. The JSONs win if the two
  ever disagree.
- ``run.log``: everything the run printed.
- ``diagnostics/*.json``: optional attachments (``--attach``), such as
  the ``benchmarks/greedy_mismatch.py`` report. Hashed, not results.

``scripts/ingest_gpu_bundle.py`` validates the zip and builds the README
tables from the JSONs. Nothing in this script writes or edits a number.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).parent.parent))

BUNDLE_FORMAT = 1

__all__ = ["BUNDLE_FORMAT", "build_bundle", "main", "render_profile"]


class _Tee(io.TextIOBase):
    """Write to several text streams at once."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_files_modified() -> bool | None:
    """Return whether tracked files differ from the commit (``None`` without git)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fmt(value: object) -> str:
    """Print a JSON value as stored: ``repr`` for floats, no rounding."""
    return repr(value) if isinstance(value, float) else str(value)


def render_profile(documents: list[dict[str, Any]]) -> str:
    """Plain-text profile of a set of result documents, values as stored.

    Parameters
    ----------
    documents : list of dict
        Result documents written by the harness.

    Returns
    -------
    str
        One block per (model, dataset, mode), FlashSpec first.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for doc in documents:
        decoding = doc.get("decoding") or {}
        mode = str(decoding.get("mode", "?"))
        if mode == "sample":
            mode = f"sample T={decoding.get('temperature')}"
        key = (str(doc.get("model")), str(doc.get("dataset")), mode)
        groups.setdefault(key, []).append(doc)
    lines: list[str] = []
    for (model, dataset, mode), docs in sorted(groups.items()):
        docs.sort(key=lambda item: (item.get("method") != "flashspec_ucb", str(item.get("method"))))
        drafter = docs[0].get("drafter", "?")
        lines.append("=" * 78)
        lines.append(f"target={model}  drafter={drafter}  dataset={dataset}  mode={mode}")
        lines.append(f"commit={docs[0].get('git_commit')}  n_samples={docs[0].get('n_samples')}")
        lines.append("-" * 78)
        for doc in docs:
            metrics = doc.get("metrics") or {}
            lines.append(f"[{doc.get('method')}]")
            for key in (
                "tokens_per_second_median",
                "tokens_per_second_p25",
                "tokens_per_second_p75",
                "tokens_per_second_mean",
                "acceptance_rate_mean",
                "gpu_memory_gb",
            ):
                if key in metrics:
                    lines.append(f"  {key} = {_fmt(metrics[key])}")
            for key in sorted(metrics):
                if key.startswith("speedup_vs_"):
                    lines.append(f"  {key} = {_fmt(metrics[key])}")
            counts = doc.get("forward_counts") or {}
            for key in sorted(counts):
                if "lengths" not in key:
                    lines.append(f"  forward_counts.{key} = {_fmt(counts[key])}")
            for section in (
                "step_profile",
                "host_syncs",
                "nonfinite_logits",
                "greedy_token_match",
                "greedy_token_match_compiled",
            ):
                payload = doc.get(section)
                if isinstance(payload, dict):
                    for key in sorted(payload):
                        lines.append(f"  {section}.{key} = {_fmt(payload[key])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_bundle(
    configs: list[Path],
    output_dir: Path,
    zip_path: Path | None = None,
    attachments: list[Path] | None = None,
) -> tuple[Path, bool]:
    """Run ``configs`` and write the bundle zip.

    Parameters
    ----------
    configs : list of Path
        Benchmark YAML files, run in order.
    output_dir : Path
        Working directory for results, manifest, profile and log.
    zip_path : Path or None
        Destination zip. Default: ``output_dir.parent /
        flashspec_gpu_bundle_<commit>.zip``.
    attachments : list of Path or None
        JSON files stored as ``diagnostics/<name>``.

    Returns
    -------
    tuple[Path, bool]
        The zip path, and whether every config finished.
    """
    from benchmarks.harness import detect_hardware, git_commit, run_config_file

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    commit = git_commit()
    started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    config_rows: list[dict[str, object]] = []
    with (
        log_path.open("w", encoding="utf-8") as log,
        contextlib.redirect_stdout(_Tee(sys.__stdout__ or sys.stdout, log)),  # type: ignore[type-var]
    ):
        print(f"FlashSpec GPU bundle: commit {commit}, {len(configs)} config(s)")
        for config in configs:
            config_started = time.perf_counter()
            row: dict[str, object] = {"path": str(config), "sha256": _sha256(config)}
            print(f"\n=== {config} ===")
            try:
                written = run_config_file(config, results_dir)
            except Exception as exc:  # noqa: BLE001 - recorded, and the bundle still ships
                message = f"{type(exc).__name__}: {exc}"
                row.update({"status": "error", "error": message, "files": []})
                print(f"ERROR in {config}: {type(exc).__name__}: {exc}")
            else:
                row.update({"status": "ok", "files": [path.name for path in written]})
            row["duration_s"] = time.perf_counter() - config_started
            print(f"=== {config}: {row['status']} in {row['duration_s']} s ===")
            config_rows.append(row)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        print(f"\nTotal: {time.perf_counter() - started} s")

    documents = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(results_dir.glob("*.json"))
    ]
    profile_path = output_dir / "profile.txt"
    profile_path.write_text(render_profile(documents), encoding="utf-8")
    files = sorted(results_dir.glob("*.json"))
    attached = {f"diagnostics/{path.name}": path for path in attachments or []}
    manifest: dict[str, object] = {
        "bundle_format": BUNDLE_FORMAT,
        "git_commit": commit,
        "tracked_files_modified": _tracked_files_modified(),
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "duration_s": time.perf_counter() - started,
        "configs": config_rows,
        "hardware": detect_hardware(),
        "versions": {
            "python": platform.python_version(),
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "flashspec": _version("flashspec"),
        },
        "files": {
            **{f"results/{path.name}": _sha256(path) for path in files},
            **{name: _sha256(path) for name, path in attached.items()},
            "profile.txt": _sha256(profile_path),
            "run.log": _sha256(log_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    destination = zip_path or output_dir.parent / f"flashspec_gpu_bundle_{commit}.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, "manifest.json")
        archive.write(profile_path, "profile.txt")
        archive.write(log_path, "run.log")
        for path in files:
            archive.write(path, f"results/{path.name}")
        for name, path in attached.items():
            archive.write(path, name)
    complete = all(row["status"] == "ok" for row in config_rows)
    return destination, complete


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns 0 when every config finished."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--attach", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    destination, complete = build_bundle(args.config, args.output_dir, args.zip, args.attach)
    print(f"Bundle written to {destination}")
    if not complete:
        print("At least one config failed; see manifest.json and run.log in the bundle.")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
