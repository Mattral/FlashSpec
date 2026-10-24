"""GPU results bundle: build, validate, ingest, and README regeneration.

The bundles here come from a CPU toy run. The acceptance test rewrites
their labels (GPU name, dataset, measurement kind) inside ``tmp_path`` to
exercise the accept path; those files are fixtures, never results, and
nothing here writes into ``benchmarks/results/``.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from benchmarks.gpu_bundle import build_bundle, render_profile
from scripts.ingest_gpu_bundle import (
    BEGIN,
    END,
    BundleError,
    ingest,
    render_tables,
    validate_bundle,
)

_TOY_YAML = """
model: toy_llama
device: cpu
gamma: 2
max_new_tokens: 4
n_warmup: 0
n_samples: 2
prompt_length: 5
vocab_size: 128
target_hidden: 32
target_layers: 1
drafter_hidden: 32
drafter_layers: 1
dataset: synthetic_prompts
temperatures:
  - 0.0
methods:
  - flashspec_ucb
  - hf_greedy
  - hf_assisted
"""


@pytest.fixture(scope="module")
def toy_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one bundle from a tiny CPU config."""
    root = tmp_path_factory.mktemp("bundle")
    config = root / "toy_modes.yaml"
    config.write_text(_TOY_YAML, encoding="utf-8")
    zip_path, complete = build_bundle([config], root / "work")
    assert complete
    return zip_path


def _rewrite(
    source: Path,
    destination: Path,
    edit_doc: Callable[[dict[str, Any]], None] | None = None,
    edit_manifest: Callable[[dict[str, Any]], None] | None = None,
    extra_member: tuple[str, bytes] | None = None,
    rehash: bool = True,
) -> Path:
    """Copy a bundle, editing documents and manifest, then fix the hashes."""
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members.pop("manifest.json"))
    for name in list(members):
        if name.startswith("results/") and edit_doc is not None:
            doc = json.loads(members[name])
            edit_doc(doc)
            members[name] = (json.dumps(doc, indent=2) + "\n").encode()
    if edit_manifest is not None:
        edit_manifest(manifest)
    if rehash:
        manifest["files"] = {
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        }
    if extra_member is not None:
        members[extra_member[0]] = extra_member[1]
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in members.items():
            archive.writestr(name, data)
    return destination


def _as_gpu_run(doc: dict[str, Any]) -> None:
    doc["measurement_kind"] = "end_to_end"
    doc["dataset"] = "mt_bench"
    doc["hardware"] = {**doc["hardware"], "gpu": "Fixture GPU"}


def _clean(manifest: dict[str, Any]) -> None:
    manifest["tracked_files_modified"] = False


def _readme(tmp_path: Path) -> Path:
    path = tmp_path / "README.md"
    path.write_text(f"# Title\n\nbefore\n\n{BEGIN}\nold\n{END}\n\nafter\n", encoding="utf-8")
    return path


class TestBuild:
    """The bundle holds the JSONs, a manifest with hashes, a profile and the log."""

    def test_contents(self, toy_bundle: Path) -> None:
        """Members, manifest fields and hashes are consistent."""
        with zipfile.ZipFile(toy_bundle) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            profile = archive.read("profile.txt").decode()
        results = {name for name in names if name.startswith("results/")}
        assert {"manifest.json", "profile.txt", "run.log"} <= names
        assert len(results) == 3
        assert manifest["bundle_format"] == 1
        assert [row["status"] for row in manifest["configs"]] == ["ok"]
        assert set(manifest["files"]) == names - {"manifest.json"}
        assert set(manifest["hardware"]) <= {
            "platform",
            "machine",
            "python",
            "torch",
            "avx2",
            "gpu",
            "cuda",
            "gpu_count",
        }
        assert "[flashspec_ucb]" in profile
        assert "step_profile.tokens_per_step_mean" in profile

    def test_failed_config_is_recorded(self, tmp_path: Path) -> None:
        """A config that cannot run is listed as an error and the bundle still ships."""
        broken = tmp_path / "cuda.yaml"
        broken.write_text(_TOY_YAML.replace("device: cpu", "device: cuda:7"), encoding="utf-8")
        zip_path, complete = build_bundle([broken], tmp_path / "work")
        assert not complete
        with zipfile.ZipFile(zip_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        assert manifest["configs"][0]["status"] == "error"


class TestValidate:
    """Only clean, hashed, GPU, real-prompt bundles at a known commit pass."""

    def test_toy_cpu_bundle_is_rejected(self, toy_bundle: Path) -> None:
        """No GPU and toy measurement kind are both named."""
        with pytest.raises(BundleError) as caught:
            validate_bundle(toy_bundle)
        text = str(caught.value)
        assert "hardware.gpu is missing" in text
        assert "measurement_kind" in text
        assert "not a vendored prompt set" in text

    def test_accepted_after_relabel(self, toy_bundle: Path, tmp_path: Path) -> None:
        """A bundle that meets every rule validates."""
        good = _rewrite(toy_bundle, tmp_path / "good.zip", _as_gpu_run, _clean)
        manifest, documents = validate_bundle(good)
        assert len(documents) == 3
        assert manifest["git_commit"]

    @pytest.mark.parametrize(
        ("label", "kwargs", "needle"),
        [
            (
                "dirty",
                {
                    "edit_doc": _as_gpu_run,
                    "edit_manifest": lambda m: m.update(tracked_files_modified=True),
                },
                "tracked files were modified",
            ),
            (
                "hash",
                {"edit_doc": _as_gpu_run, "edit_manifest": _clean, "rehash": False},
                "sha256 does not match",
            ),
            (
                "commit",
                {
                    "edit_doc": lambda d: (_as_gpu_run(d), d.update(git_commit="0000000")),
                    "edit_manifest": _clean,
                },
                "!= manifest",
            ),
            (
                "hardware",
                {
                    "edit_doc": lambda d: (_as_gpu_run(d), d["hardware"].update(processor="x")),
                    "edit_manifest": _clean,
                },
                "outside the allowed set",
            ),
            (
                "unknown_commit",
                {
                    "edit_doc": lambda d: (_as_gpu_run(d), d.update(git_commit="fffffff")),
                    "edit_manifest": lambda m: (_clean(m), m.update(git_commit="fffffff")),
                },
                "not in this repository",
            ),
            (
                "missing_section",
                {
                    "edit_doc": lambda d: (_as_gpu_run(d), d.pop("nonfinite_logits")),
                    "edit_manifest": _clean,
                },
                "missing section 'nonfinite_logits'",
            ),
            (
                "missing_outputs",
                {
                    "edit_doc": lambda d: (_as_gpu_run(d), d.pop("outputs")),
                    "edit_manifest": _clean,
                },
                "missing section 'outputs'",
            ),
            (
                "tampered_outputs",
                {
                    "edit_doc": lambda d: (
                        _as_gpu_run(d),
                        d["outputs"]["prompts"][0]["token_ids"].append(7),
                    ),
                    "edit_manifest": _clean,
                },
                "sha256 or n_tokens does not match",
            ),
            (
                "short_outputs",
                {
                    "edit_doc": lambda d: (_as_gpu_run(d), d["outputs"]["prompts"].pop()),
                    "edit_manifest": _clean,
                },
                "outputs for n_samples",
            ),
        ],
    )
    def test_rejections(
        self, toy_bundle: Path, tmp_path: Path, label: str, kwargs: dict[str, Any], needle: str
    ) -> None:
        """Each rule rejects with a message that says why."""
        bad = _rewrite(toy_bundle, tmp_path / f"{label}.zip", **kwargs)
        with pytest.raises(BundleError, match=needle):
            validate_bundle(bad)

    def test_path_traversal_member_is_rejected(self, toy_bundle: Path, tmp_path: Path) -> None:
        """Members outside the fixed layout are refused before anything is read."""
        evil = ("../x.json", b"{}")
        bad = _rewrite(toy_bundle, tmp_path / "evil.zip", _as_gpu_run, _clean, extra_member=evil)
        with pytest.raises(BundleError, match="unexpected members"):
            validate_bundle(bad)


class TestIngest:
    """Accepted bundles are unpacked and the README block is rebuilt from JSON only."""

    def test_ingest_writes_dest_and_readme(self, toy_bundle: Path, tmp_path: Path) -> None:
        """Values in the README are the stored values; the rest of the README is untouched."""
        good = _rewrite(toy_bundle, tmp_path / "good.zip", _as_gpu_run, _clean)
        readme = _readme(tmp_path)
        dest = tmp_path / "gpu"
        body = ingest([good], destination=dest, readme=readme)
        docs = [json.loads(path.read_text()) for path in sorted(dest.glob("*/results/*.json"))]
        assert len(docs) == 3
        text = readme.read_text()
        assert text.startswith("# Title\n\nbefore\n\n")
        assert text.endswith("\n\nafter\n")
        assert "old" not in text
        spec = next(doc for doc in docs if doc["method"] == "flashspec_ucb")
        assert repr(spec["metrics"]["tokens_per_second_median"]) in body
        assert "Fixture GPU" in body
        assert "not rounded" in body

    def test_dry_run_writes_nothing_and_sig_formats(self, toy_bundle: Path, tmp_path: Path) -> None:
        """A dry run returns the table; --sig changes only the display."""
        good = _rewrite(toy_bundle, tmp_path / "good.zip", _as_gpu_run, _clean)
        readme = _readme(tmp_path)
        before = readme.read_text()
        body = ingest([good], destination=tmp_path / "gpu", readme=readme, dry_run=True, sig=3)
        assert readme.read_text() == before
        assert not (tmp_path / "gpu").exists()
        assert "3 significant figures" in body

    def test_rejected_bundle_writes_nothing(self, toy_bundle: Path, tmp_path: Path) -> None:
        """One bad bundle stops the whole ingest before any write."""
        readme = _readme(tmp_path)
        before = readme.read_text()
        with pytest.raises(BundleError):
            ingest([toy_bundle], destination=tmp_path / "gpu", readme=readme)
        assert readme.read_text() == before
        assert not (tmp_path / "gpu").exists()


def test_render_helpers_handle_empty_input() -> None:
    """No documents gives the placeholder, and the profile renders an empty set."""
    assert render_tables([]) == "No GPU bundle has been ingested yet.\n"
    assert render_profile([]) == "\n"


def test_repository_readme_has_the_block() -> None:
    """README.md carries the generated block markers."""
    text = Path("README.md").read_text(encoding="utf-8")
    assert BEGIN in text
    assert END in text


def test_dry_run_of_an_ingested_commit_shows_it_once(toy_bundle: Path, tmp_path: Path) -> None:
    """Re-validating a bundle that is already unpacked does not duplicate its rows."""
    good = _rewrite(toy_bundle, tmp_path / "good.zip", _as_gpu_run, _clean)
    readme = _readme(tmp_path)
    dest = tmp_path / "gpu"
    first = ingest([good], destination=dest, readme=readme)
    again = ingest([good], destination=dest, readme=readme, dry_run=True)
    assert again == first


def test_attachments_are_bundled_and_accepted(tmp_path: Path) -> None:
    """A diagnostics JSON rides along, is hashed, and passes the layout check."""
    config = tmp_path / "toy.yaml"
    config.write_text(_TOY_YAML, encoding="utf-8")
    note = tmp_path / "greedy_mismatch.json"
    note.write_text('{"cases": []}\n', encoding="utf-8")
    zip_path, complete = build_bundle([config], tmp_path / "work", attachments=[note])
    assert complete
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.read("diagnostics/greedy_mismatch.json") == note.read_bytes()
    assert "diagnostics/greedy_mismatch.json" in manifest["files"]
    good = _rewrite(zip_path, tmp_path / "good.zip", _as_gpu_run, _clean)
    _manifest, documents = validate_bundle(good)
    assert all(name.startswith("results/") for name in documents)


def test_render_status_needs_compiled_results() -> None:
    """The README status table is only built from flashspec_compiled results."""
    from scripts.ingest_gpu_bundle import render_status

    assert render_status([]) is None
    assert render_status([{"method": "flashspec_ucb"}]) is None


def test_render_status_shows_both_greedy_matches() -> None:
    """Compiled-vs-compiled and compiled-vs-eager matches get their own columns."""
    from scripts.ingest_gpu_bundle import render_status

    doc = {
        "method": "flashspec_compiled",
        "git_commit": "abc1234",
        "timestamp": "t",
        "dataset": "humaneval",
        "decoding": {"mode": "greedy", "temperature": 0.0, "max_new_tokens": 8},
        "metrics": {},
        "greedy_token_match_compiled": {"prompts_matched": 3, "n_prompts": 3},
        "greedy_token_match": {"prompts_matched": 2, "n_prompts": 3},
    }
    table = render_status([doc])
    assert table is not None
    header, _rule, row = table.splitlines()[2:5]
    assert "greedy match vs compiled HF | greedy match vs eager HF" in header
    assert row.endswith("| 3/3 | 2/3 |")
    older = {key: value for key, value in doc.items() if key != "greedy_token_match_compiled"}
    assert render_status([older]).splitlines()[4].endswith("| n/a | 2/3 |")  # type: ignore[union-attr]


def test_greedy_compiled_needs_the_compiled_match() -> None:
    """A greedy flashspec_compiled run next to hf_greedy_compiled must say if they match."""
    from scripts.ingest_gpu_bundle import _check_document

    doc = {
        "method": "flashspec_compiled",
        "decoding": {"mode": "greedy"},
        "metrics": {"speedup_vs_hf_greedy_compiled_median": 1.0},
        "greedy_token_match": {},
    }
    assert any("greedy_token_match_compiled" in e for e in _check_document("r", doc, "c"))
    doc["greedy_token_match_compiled"] = {}
    assert not any("greedy_token_match_compiled" in e for e in _check_document("r", doc, "c"))
