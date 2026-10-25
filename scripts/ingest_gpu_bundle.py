"""Validate a GPU results bundle and rebuild the README GPU tables from its JSONs.

Usage::

    python scripts/ingest_gpu_bundle.py                 # every zip in benchmarks/results/incoming/
    python scripts/ingest_gpu_bundle.py path/to/bundle.zip
    python scripts/ingest_gpu_bundle.py --dry-run       # validate and print, write nothing
    python scripts/ingest_gpu_bundle.py --sig 3         # README cells at 3 significant figures
    python scripts/ingest_gpu_bundle.py --readme-commit 8d036d7   # README from one bundle only

A bundle is the zip ``benchmarks/gpu_bundle.py`` writes. It is accepted
only when:

- it holds nothing but ``manifest.json``, ``profile.txt``, ``run.log``,
  ``results/<name>.json`` and ``diagnostics/<name>.json``, and every
  sha256 in the manifest matches;
- the manifest commit exists in this repository, tracked files were not
  modified at run time, and every config finished (``--allow-partial``
  relaxes the last);
- every result passes ``scripts/update_benchmark_results.py`` validation,
  records the manifest commit, is an ``end_to_end`` run on a GPU with real
  prompts, carries the phase 4 sections and per-prompt ``outputs`` (token
  ids whose sha256 matches the stored hash), a greedy ``flashspec_compiled``
  run next to ``hf_greedy_compiled`` carries ``greedy_token_match_compiled``,
  and its ``hardware`` has only the
  harness fields (platform, machine, python, torch, avx2, gpu, cuda,
  gpu_count).

Accepted bundles are unpacked to ``benchmarks/results/gpu/<commit>/``.
The README block between the ``gpu-results`` markers is then regenerated
from every JSON under ``benchmarks/results/gpu/``, or, with
``--readme-commit``, from that one bundle's JSONs only (the release
tables); the other bundles stay on disk as history. Cells are the stored
values; ``--sig N`` formats them to N significant figures for display,
and the JSON keeps full precision either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.prompt_sets import REAL_DATASETS  # noqa: E402
from scripts.update_benchmark_results import _validate_result_file  # noqa: E402

INCOMING = Path("benchmarks/results/incoming")
DESTINATION = Path("benchmarks/results/gpu")
README = Path("README.md")
BEGIN = "<!-- BEGIN GENERATED: gpu-results (scripts/ingest_gpu_bundle.py) -->"
END = "<!-- END GENERATED: gpu-results -->"
STATUS_BEGIN = "<!-- BEGIN GENERATED: status (scripts/ingest_gpu_bundle.py) -->"
STATUS_END = "<!-- END GENERATED: status -->"
ALLOWED_HARDWARE = frozenset(
    {"platform", "machine", "python", "torch", "avx2", "gpu", "cuda", "gpu_count"}
)
SPEC = "flashspec_ucb"
SPECS = ("flashspec_ucb", "flashspec_compiled")
BASELINES = (
    "hf_greedy",
    "hf_greedy_compiled",
    "hf_sample",
    "hf_sample_compiled",
    "hf_assisted",
    "vanilla_ar",
)

__all__ = ["BEGIN", "END", "ingest", "main", "render_tables", "validate_bundle"]


class BundleError(ValueError):
    """The bundle is not accepted. The message lists every problem found."""


def _commit_exists(commit: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _member_ok(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    if name in {"manifest.json", "profile.txt", "run.log"}:
        return True
    if len(path.parts) == 2 and path.parts[0] == "logs":
        return path.suffix == ".log"
    return (
        len(path.parts) == 2
        and path.parts[0] in {"results", "diagnostics"}
        and path.suffix == ".json"
    )


def _check_document(name: str, doc: dict[str, Any], commit: str) -> list[str]:
    errors: list[str] = []
    if doc.get("status") != "measured":
        errors.append(f"{name}: status is {doc.get('status')!r}, not 'measured'")
    if doc.get("git_commit") != commit:
        errors.append(f"{name}: git_commit {doc.get('git_commit')!r} != manifest {commit!r}")
    if doc.get("measurement_kind") != "end_to_end":
        errors.append(f"{name}: measurement_kind {doc.get('measurement_kind')!r} != 'end_to_end'")
    hardware = doc.get("hardware")
    if not isinstance(hardware, dict):
        errors.append(f"{name}: hardware is missing")
    else:
        extra = sorted(set(hardware) - ALLOWED_HARDWARE)
        if extra:
            errors.append(f"{name}: hardware has fields outside the allowed set: {extra}")
        if not hardware.get("gpu"):
            errors.append(f"{name}: hardware.gpu is missing; this is not a GPU run")
    if doc.get("dataset") not in REAL_DATASETS:
        errors.append(f"{name}: dataset {doc.get('dataset')!r} is not a vendored prompt set")
    for section in (
        "decoding",
        "drafter",
        "prompt_format",
        "forward_counts",
        "host_syncs",
        "nonfinite_logits",
        "outputs",
    ):
        if section not in doc:
            errors.append(f"{name}: missing section {section!r}")
    errors.extend(_check_outputs(name, doc))
    method = doc.get("method")
    if method in SPECS and "step_profile" not in doc:
        errors.append(f"{name}: FlashSpec result without step_profile")
    decoding = doc.get("decoding") or {}
    if decoding.get("mode") == "greedy" and method in (*SPECS, "hf_assisted"):
        if "greedy_token_match" not in doc:
            errors.append(f"{name}: greedy {method} result without greedy_token_match")
    if "phase6" in doc:
        errors.extend(_check_phase6(name, doc))
    metrics = doc.get("metrics") or {}
    if (
        decoding.get("mode") == "greedy"
        and method == "flashspec_compiled"
        and "speedup_vs_hf_greedy_compiled_median" in metrics
        and "greedy_token_match_compiled" not in doc
    ):
        errors.append(f"{name}: greedy flashspec_compiled without greedy_token_match_compiled")
    return errors


def _check_phase6(name: str, doc: dict[str, Any]) -> list[str]:
    """Extra sections a Phase 6 (``benchmarks/phase6.py``) document must carry."""
    errors: list[str] = []
    metrics = doc.get("metrics") or {}
    if "workload_tokens_per_second" not in metrics:
        errors.append(f"{name}: phase6 result without metrics.workload_tokens_per_second")
    if "arms" not in doc:
        return errors  # a HuggingFace baseline
    for section in ("arm_report", "step_profile", "compile", "cuda_graphs", "bandit"):
        if section not in doc:
            errors.append(f"{name}: phase6 FlashSpec result without {section!r}")
    if (doc.get("decoding") or {}).get("mode") == "greedy":
        for section in ("greedy_token_match", "greedy_token_match_compiled"):
            if section not in doc:
                errors.append(f"{name}: greedy phase6 FlashSpec result without {section!r}")
    return errors


def _check_blocks(
    manifest: dict[str, Any], documents: dict[str, dict[str, Any]], members: set[str]
) -> list[str]:
    """Check a bundle assembled across sessions: each block and document traces to one session.

    Checked when the manifest has ``blocks`` (``gpu_bundle.py --checkpoint-dir``):
    each block names the bundle's commit, a config sha256 the manifest lists,
    a clean tree, a listed session with its log in the zip, and files whose
    sha256 match the manifest; every phase6 document is in exactly one
    block and names that block's session.
    """
    errors: list[str] = []
    commit = manifest.get("git_commit")
    config_hashes = {row.get("sha256") for row in manifest.get("configs", [])}
    sessions = {str(item.get("id")): item for item in manifest.get("sessions") or []}
    listed = manifest.get("files") or {}
    owner: dict[str, dict[str, Any]] = {}
    for block in manifest.get("blocks") or []:
        label = f"block {block.get('block')} (session {block.get('session')})"
        if block.get("git_commit") != commit:
            errors.append(f"{label}: commit {block.get('git_commit')!r} != {commit!r}")
        if block.get("config_sha256") not in config_hashes:
            errors.append(f"{label}: config sha256 is not the config in this manifest")
        if block.get("tracked_files_modified") is not False:
            modified = block.get("tracked_files_modified")
            errors.append(f"{label}: tracked files modified: {modified!r}")
        session = str(block.get("session"))
        if session not in sessions:
            errors.append(f"{label}: session is not in manifest.sessions")
        elif f"logs/{session}.log" not in members:
            errors.append(f"{label}: logs/{session}.log is not in the zip")
        for file_name, digest in (block.get("files") or {}).items():
            member = f"results/{file_name}"
            if listed.get(member) != digest:
                errors.append(f"{label}: {member} sha256 differs from the block record")
            if member in owner:
                errors.append(f"{member} is in two blocks")
            owner[member] = block
    for member, doc in documents.items():
        if "phase6" not in doc:
            continue
        block = owner.get(member)
        if block is None:
            errors.append(f"{member}: phase6 result is in no block of manifest.blocks")
        elif (doc.get("phase6") or {}).get("session") != block.get("session"):
            errors.append(f"{member}: session differs from its block's session")
    return errors


def _check_outputs(name: str, doc: dict[str, Any]) -> list[str]:
    """Per-prompt outputs: one entry per sample, and each sha256 matches its token ids."""
    outputs = doc.get("outputs")
    if outputs is None:
        return []
    prompts = outputs.get("prompts") if isinstance(outputs, dict) else None
    if not isinstance(prompts, list):
        return [f"{name}: outputs.prompts is not a list"]
    errors: list[str] = []
    if len(prompts) != doc.get("n_samples"):
        errors.append(f"{name}: {len(prompts)} outputs for n_samples={doc.get('n_samples')!r}")
    for index, entry in enumerate(prompts):
        tokens = entry.get("token_ids") if isinstance(entry, dict) else None
        if not isinstance(tokens, list):
            errors.append(f"{name}: outputs.prompts[{index}] has no token_ids")
            continue
        digest = hashlib.sha256(",".join(str(int(t)) for t in tokens).encode("ascii")).hexdigest()
        if entry.get("sha256") != digest or entry.get("n_tokens") != len(tokens):
            errors.append(f"{name}: outputs.prompts[{index}] sha256 or n_tokens does not match")
    return errors


def validate_bundle(
    zip_path: Path, *, allow_partial: bool = False
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Check one bundle and return its manifest and result documents.

    Raises
    ------
    BundleError
        Listing every problem, when the bundle is not accepted.
    """
    errors: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        bad = [name for name in names if not _member_ok(name)]
        if bad:
            raise BundleError(f"{zip_path.name}: unexpected members {bad}")
        if "manifest.json" not in names:
            raise BundleError(f"{zip_path.name}: no manifest.json")
        manifest = json.loads(archive.read("manifest.json"))
        payloads = {name: archive.read(name) for name in names if name != "manifest.json"}
    if manifest.get("bundle_format") != 1:
        errors.append(f"bundle_format {manifest.get('bundle_format')!r} is not 1")
    commit = str(manifest.get("git_commit", ""))
    if not commit or commit == "unknown":
        errors.append("manifest has no git_commit")
    elif not _commit_exists(commit):
        errors.append(f"commit {commit} is not in this repository (git fetch, then retry)")
    if manifest.get("tracked_files_modified") is not False:
        errors.append(
            "tracked files were modified (or unknown) when the bundle ran: "
            f"{manifest.get('tracked_files_modified')!r}"
        )
    failed = [row for row in manifest.get("configs", []) if row.get("status") != "ok"]
    if failed and not allow_partial:
        unfinished = [(row["path"], row.get("error")) for row in failed]
        errors.append(f"configs did not finish: {unfinished}")
    listed = manifest.get("files") or {}
    if set(listed) != set(payloads):
        errors.append(
            f"manifest files {sorted(listed)} do not match the zip members {sorted(payloads)}"
        )
    for name, data in payloads.items():
        if name in listed and hashlib.sha256(data).hexdigest() != listed[name]:
            errors.append(f"{name}: sha256 does not match the manifest")
    documents: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as scratch_dir:
        for name, data in sorted(payloads.items()):
            if not name.startswith("results/"):
                continue
            doc = json.loads(data)
            documents[name] = doc
            # The schema validator reads a file; give it this one under its own name.
            scratch = Path(scratch_dir) / PurePosixPath(name).name
            scratch.write_bytes(data)
            errors.extend(line.strip() for line in _validate_result_file(scratch))
            errors.extend(_check_document(name, doc, commit))
    if not documents:
        errors.append("the bundle has no result JSON")
    if manifest.get("blocks") is not None:
        errors.extend(_check_blocks(manifest, documents, set(payloads)))
    if errors:
        raise BundleError(f"{zip_path.name} rejected:\n  - " + "\n  - ".join(errors))
    return manifest, documents


def _cell(value: object, sig: int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{sig}g}" if sig is not None else repr(value)
    return str(value)


def render_tables(documents: list[dict[str, Any]], *, sig: int | None = None) -> str:
    """Markdown for the README block, built only from result documents.

    One table per (target, drafter); one row per dataset, mode and method.
    """
    if not documents:
        return "No GPU bundle has been ingested yet.\n"
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for doc in documents:
        pairs.setdefault((str(doc.get("model")), str(doc.get("drafter"))), []).append(doc)
    method_rank = {
        **{name: index for index, name in enumerate(SPECS)},
        **{name: index + len(SPECS) for index, name in enumerate(BASELINES)},
    }
    lines: list[str] = []
    precision = (
        f"Cells are shown to {sig} significant figures; the JSON keeps full precision."
        if sig is not None
        else "Cells are the stored JSON values, not rounded."
    )
    for (model, drafter), docs in sorted(pairs.items()):
        first = docs[0]
        hardware = first.get("hardware") or {}
        commits = sorted({str(doc.get("git_commit")) for doc in docs})
        lines.append(f"#### `{model}` verified, `{drafter}` drafting")
        lines.append("")
        lines.append(
            f"GPU {hardware.get('gpu')} · torch {hardware.get('torch')} · "
            f"gamma {first.get('gamma')} · {len(docs)} result files · git "
            + ", ".join(f"`{commit}`" for commit in commits)
            + f". Prompts: {first.get('prompt_format')}; n per row: {first.get('n_samples')}. "
            + precision
        )
        lines.append("")
        lines.append(
            "| dataset | mode | method | tok/s median | tok/s p25 | tok/s p75 | acceptance "
            "| tokens / target fwd | target fwd / token | speedup vs plain HF (median) "
            "| speedup vs compiled plain HF (median) | speedup vs hf_assisted (median) "
            "| c (step profile) | greedy match | NaN/inf logits | peak GB |"
        )
        lines.append("|" + "---|" * 16)

        def _key(doc: dict[str, Any]) -> tuple[str, str, int]:
            decoding = doc.get("decoding") or {}
            return (
                str(doc.get("dataset")),
                f"{decoding.get('mode')}{decoding.get('temperature')}",
                method_rank.get(str(doc.get("method")), 99),
            )

        for doc in sorted(docs, key=_key):
            metrics = doc.get("metrics") or {}
            decoding = doc.get("decoding") or {}
            mode = (
                "greedy" if decoding.get("mode") == "greedy" else f"T={decoding.get('temperature')}"
            )
            counts = doc.get("forward_counts") or {}
            plain = "hf_greedy" if decoding.get("mode") == "greedy" else "hf_sample"
            match_cell = _fraction(doc.get("greedy_token_match"))
            nonfinite = doc.get("nonfinite_logits") or {}
            bad = None
            if "target_entries" in nonfinite:
                bad = int(nonfinite.get("target_entries", 0)) + int(
                    nonfinite.get("drafter_entries", 0)
                )
            row = [
                str(doc.get("dataset")),
                mode,
                str(doc.get("method")),
                _cell(metrics.get("tokens_per_second_median"), sig),
                _cell(metrics.get("tokens_per_second_p25"), sig),
                _cell(metrics.get("tokens_per_second_p75"), sig),
                _cell(metrics.get("acceptance_rate_mean"), sig),
                _cell(counts.get("emitted_tokens_per_target_forward_after_prefill"), sig),
                _cell(counts.get("target_forwards_per_emitted_token"), sig),
                _cell(metrics.get(f"speedup_vs_{plain}_median"), sig),
                _cell(metrics.get(f"speedup_vs_{plain}_compiled_median"), sig),
                _cell(metrics.get("speedup_vs_hf_assisted_median"), sig),
                _cell((doc.get("step_cost") or {}).get("c"), sig),
                match_cell,
                _cell(bad, sig),
                _cell(metrics.get("gpu_memory_gb"), sig),
            ]
            lines.append("| " + " | ".join(row) + " |")
        costs = sorted(
            {
                (
                    str(doc.get("dataset")),
                    json.dumps((doc.get("forward_cost") or {}).get("eager")),
                    json.dumps((doc.get("forward_cost") or {}).get("compiled")),
                )
                for doc in docs
                if doc.get("forward_cost")
            }
        )
        for dataset, eager, compiled in costs:
            eager_c = (json.loads(eager) or {}).get("c")
            compiled_c = (json.loads(compiled) or {}).get("c") if compiled != "null" else None
            lines.append(
                f"- forward_cost ({dataset}): c eager {_cell(eager_c, sig)}, "
                f"c compiled {_cell(compiled_c, sig)}"
            )
        lines.append("")
    lines.append(
        "Speedups are medians of per-prompt ratios against the method in the same run. "
        "NaN/inf counts come from one untimed probe generation per method on the first prompt."
    )
    return "\n".join(lines) + "\n"


def render_status(documents: list[dict[str, Any]], *, sig: int | None = None) -> str | None:
    """Headline table: compiled FlashSpec vs compiled HF, eager HF and assisted HF.

    Built from the newest commit that has ``flashspec_compiled`` results;
    ``None`` when there is none.
    """
    compiled = [doc for doc in documents if doc.get("method") == "flashspec_compiled"]
    if not compiled:
        return None
    commit = max(compiled, key=lambda doc: str(doc.get("timestamp")))["git_commit"]
    docs = [doc for doc in compiled if doc.get("git_commit") == commit]
    others = {
        (doc.get("dataset"), (doc.get("decoding") or {}).get("mode"), doc.get("method")): doc
        for doc in documents
        if doc.get("git_commit") == commit
    }
    first = docs[0]
    hardware = first.get("hardware") or {}
    lines = [
        f"Measured at git `{commit}`: `{first.get('model')}` verified by "
        f"`{first.get('drafter')}`, {hardware.get('gpu')}, fp16, batch 1, gamma "
        f"{first.get('gamma')}, {first.get('n_samples')} prompts per set (chat template), "
        f"{(first.get('decoding') or {}).get('max_new_tokens')} new tokens. One model pair, "
        "one GPU. Speedups are medians of per-prompt ratios within the same run. The "
        "headline baseline is HuggingFace `generate` with a static cache compiled the same "
        "way (CUDA graphs); eager HuggingFace and assisted generation are context. "
        + (
            f"Cells are shown to {sig} significant figures."
            if sig is not None
            else "Cells are the stored JSON values."
        ),
        "",
        "| dataset | mode | FlashSpec tok/s | compiled HF tok/s | **vs compiled HF** "
        "| eager HF tok/s | vs eager HF | vs HF assisted | acceptance | c (compiled) "
        "| greedy match vs compiled HF | greedy match vs eager HF |",
        "|" + "---|" * 12,
    ]
    for doc in sorted(
        docs, key=lambda d: (str(d.get("dataset")), str((d.get("decoding") or {}).get("mode")))
    ):
        decoding = doc.get("decoding") or {}
        mode = decoding.get("mode")
        plain = "hf_greedy" if mode == "greedy" else "hf_sample"
        metrics = doc.get("metrics") or {}
        dataset = doc.get("dataset")

        def _tps(method: str, dataset: Any = dataset, mode: Any = mode) -> object:  # noqa: ANN401
            other = others.get((dataset, mode, method)) or {}
            return (other.get("metrics") or {}).get("tokens_per_second_median")

        cost = (doc.get("forward_cost") or {}).get("compiled") or {}
        match = doc.get("greedy_token_match")
        match_compiled = doc.get("greedy_token_match_compiled")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(doc.get("dataset")),
                    "greedy" if mode == "greedy" else f"T={decoding.get('temperature')}",
                    _cell(metrics.get("tokens_per_second_median"), sig),
                    _cell(_tps(f"{plain}_compiled"), sig),
                    "**" + _cell(metrics.get(f"speedup_vs_{plain}_compiled_median"), sig) + "**",
                    _cell(_tps(plain), sig),
                    _cell(metrics.get(f"speedup_vs_{plain}_median"), sig),
                    _cell(metrics.get("speedup_vs_hf_assisted_median"), sig),
                    _cell(metrics.get("acceptance_rate_mean"), sig),
                    _cell(cost.get("c"), sig),
                    _fraction(match_compiled),
                    _fraction(match),
                ]
            )
            + " |"
        )
    lines.extend(_status_notes(docs))
    return "\n".join(lines) + "\n"


def _status_notes(docs: list[dict[str, Any]]) -> list[str]:
    """Sentences under the status table, built from the documents only."""
    notes: list[str] = []
    greedy = sorted(
        (doc for doc in docs if isinstance(doc.get("greedy_token_match_compiled"), dict)),
        key=lambda doc: str(doc.get("dataset")),
    )
    if greedy:
        parts = [
            f"{doc['greedy_token_match_compiled']['prompts_matched']}/"
            f"{doc['greedy_token_match_compiled']['n_prompts']} {doc.get('dataset')} prompts"
            for doc in greedy
        ]
        notes.append(
            "Greedy: compiled FlashSpec produced the same tokens as `hf_greedy_compiled`, "
            "token for token (per-prompt output hashes are in the JSON), on "
            + " and ".join(parts)
            + "."
        )
    cold = [doc for doc in docs if isinstance(doc.get("cold_start"), dict)]
    if cold:
        fresh = [doc for doc in cold if doc["cold_start"].get("first_in_process")]
        first = min(fresh or cold, key=lambda doc: str(doc.get("timestamp")))
        record = first["cold_start"]
        notes.append(
            "Cold start: the first compiled FlashSpec call in this run "
            f"({first.get('dataset')}, {(first.get('decoding') or {}).get('mode')}) took "
            f"{record.get('first_call_ms')} ms including compilation and CUDA-graph capture."
        )
    else:
        notes.append(
            "Cold start: not recorded in this bundle. The first compiled call compiles the "
            "model and captures CUDA graphs, which can take minutes; the timed numbers above "
            "exclude it (warmup prompts absorb it)."
        )
    return [""] + notes if notes else []


def _fraction(match: object) -> str:
    """``matched/n`` from a greedy_token_match section; ``n/a`` when absent."""
    if isinstance(match, dict):
        return f"{match['prompts_matched']}/{match['n_prompts']}"
    return "n/a"


def _replace_block(readme: str, body: str, begin: str = BEGIN, end: str = END) -> str:
    if begin not in readme or end not in readme:
        raise BundleError(f"README has no {begin} ... {end} block")
    head, rest = readme.split(begin, 1)
    _old, tail = rest.split(end, 1)
    return f"{head}{begin}\n{body}{end}{tail}"


def _readme_documents(documents: list[dict[str, Any]], commit: str | None) -> list[dict[str, Any]]:
    """All documents, or only ``commit``'s; refuse a commit with none.

    Phase 6 documents (``"phase6"`` key) never enter these tables: their
    methods and blocks do not fit the columns. ``scripts/phase6_report.py``
    reports them.
    """
    documents = [doc for doc in documents if "phase6" not in doc]
    if commit is None:
        return documents
    chosen = [doc for doc in documents if str(doc.get("git_commit")) == commit]
    if not chosen:
        raise BundleError(f"no ingested result has git_commit {commit!r}")
    return chosen


def _all_documents(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/results/*.json"))
    ]


def ingest(
    zips: list[Path],
    *,
    destination: Path = DESTINATION,
    readme: Path = README,
    dry_run: bool = False,
    allow_partial: bool = False,
    sig: int | None = None,
    readme_commit: str | None = None,
) -> str:
    """Validate ``zips``, unpack the accepted ones, and regenerate the README block.

    ``readme_commit`` limits the README blocks to that bundle's documents.

    Returns
    -------
    str
        The markdown written into the README block (or that would be, on a dry run).

    Raises
    ------
    BundleError
        If any bundle is rejected. Nothing is written in that case.
    """
    accepted = [(path, *validate_bundle(path, allow_partial=allow_partial)) for path in zips]
    if dry_run:
        # A re-ingested commit replaces its unpacked copy, so show it once.
        incoming = {str(manifest["git_commit"]) for _path, manifest, _docs in accepted}
        documents = [
            doc for doc in _all_documents(destination) if str(doc.get("git_commit")) not in incoming
        ] + [doc for _path, _manifest, docs in accepted for doc in docs.values()]
        return render_tables(_readme_documents(documents, readme_commit), sig=sig)
    for path, manifest, _docs in accepted:
        target = destination / str(manifest["git_commit"])
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                out = target / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(archive.read(name))
        print(f"Accepted {path.name} -> {target}")
    documents = _readme_documents(_all_documents(destination), readme_commit)
    body = render_tables(documents, sig=sig)
    text = _replace_block(readme.read_text(encoding="utf-8"), body)
    status = render_status(documents, sig=sig)
    if status is not None and STATUS_BEGIN in text:
        text = _replace_block(text, status, STATUS_BEGIN, STATUS_END)
    readme.write_text(text, encoding="utf-8")
    print(f"Regenerated the gpu-results block in {readme}")
    return body


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundles", nargs="*", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--sig", type=int, default=None)
    parser.add_argument(
        "--readme-commit",
        default=None,
        help="build the README blocks from this bundle commit only",
    )
    args = parser.parse_args(argv)
    zips = list(args.bundles) or sorted(INCOMING.glob("*.zip"))
    if not zips:
        print(f"No bundle given and none in {INCOMING}/.")
        return 1
    try:
        body = ingest(
            zips,
            dry_run=args.dry_run,
            allow_partial=args.allow_partial,
            sig=args.sig,
            readme_commit=args.readme_commit,
        )
    except BundleError as exc:
        print(exc)
        return 2
    if args.dry_run:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
