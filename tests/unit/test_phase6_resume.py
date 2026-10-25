"""Resumable Phase 6 bundles: kill after one block, rerun, assemble, ingest-validate.

The CPU smoke config (``benchmarks/configs/tiny_llama_cpu_phase6.yaml``,
8 blocks) stands in for the T4 run. Session A is killed right after its
first block is saved (a ``KeyboardInterrupt``, which ``build_bundle`` does
not catch, like a lost Colab VM). Session B runs with the same checkpoint
folder in a fresh output folder, as a reconnected notebook would.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import benchmarks.checkpoint as checkpoint_module
import benchmarks.gpu_bundle as gpu_bundle
import benchmarks.phase6 as phase6
from benchmarks.checkpoint import BlockCheckpoint, sha256_file
from scripts.ingest_gpu_bundle import BundleError, _check_blocks, validate_bundle

_SMOKE = Path(__file__).resolve().parents[2] / "benchmarks/configs/tiny_llama_cpu_phase6.yaml"
_N_BLOCKS = 8


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _zip_contents(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], set[str]]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        docs = {
            name: json.loads(archive.read(name)) for name in names if name.startswith("results/")
        }
    return manifest, docs, names


@pytest.fixture(scope="module")
def resumed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Session A (killed after block 1), then session B to completion."""
    root = tmp_path_factory.mktemp("resume")
    checkpoint_dir = root / "drive" / "commit"
    patch = pytest.MonkeyPatch()
    runs: list[tuple[str, str]] = []
    original_run = phase6._run_block

    def _counting(**kwargs: Any) -> Any:  # noqa: ANN401
        runs.append((str(checkpoint_module.SESSION_ID), str(kwargs["dataset"])))
        return original_run(**kwargs)

    patch.setattr(phase6, "_run_block", _counting)
    # The working tree under test may be dirty; record it as clean, as on Colab.
    patch.setattr(gpu_bundle, "_tracked_files_modified", lambda: False)
    original_save = BlockCheckpoint.save
    saves = {"n": 0}

    def _save_then_die(self: BlockCheckpoint, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        meta = original_save(self, *args, **kwargs)
        saves["n"] += 1
        if saves["n"] == 1:
            raise KeyboardInterrupt("simulated disconnect after block 1")
        return meta

    try:
        patch.setattr(checkpoint_module, "SESSION_ID", "20261025T000000Z-sessiona")
        patch.setattr(BlockCheckpoint, "save", _save_then_die)
        with pytest.raises(KeyboardInterrupt):
            gpu_bundle.build_bundle([_SMOKE], root / "vm_a" / "out", checkpoint_dir=checkpoint_dir)
        after_a = sorted(path.name for path in (checkpoint_dir / "blocks").iterdir())
        patch.setattr(BlockCheckpoint, "save", original_save)
        patch.setattr(checkpoint_module, "SESSION_ID", "20261025T010000Z-sessionb")
        zip_path, complete = gpu_bundle.build_bundle(
            [_SMOKE], root / "vm_b" / "out", checkpoint_dir=checkpoint_dir
        )
    finally:
        patch.undo()
    return {
        "zip": zip_path,
        "complete": complete,
        "runs": runs,
        "after_a": after_a,
        "checkpoint_dir": checkpoint_dir,
        "log_b": (root / "vm_b" / "out" / "run.log").read_text(encoding="utf-8"),
    }


def test_only_missing_blocks_run_again(resumed: dict[str, Any]) -> None:
    """Session A saved block 1 and died; session B ran the other 7 and restored block 1."""
    assert resumed["after_a"] == ["humaneval_greedy"]
    sessions = [session for session, _dataset in resumed["runs"]]
    assert sessions.count("20261025T000000Z-sessiona") == 1
    assert sessions.count("20261025T010000Z-sessionb") == _N_BLOCKS - 1
    assert resumed["complete"] is True
    log = resumed["log_b"]
    assert "8 blocks: 1 restored from the checkpoint, 7 to run" in log
    assert "block 1/8 humaneval_greedy: restored from the checkpoint" in log
    assert "ETA about" in log
    assert (resumed["checkpoint_dir"] / resumed["zip"].name).is_file()


def test_manifest_records_sessions_and_ingest_accepts_it(resumed: dict[str, Any]) -> None:
    """Each block names its session; the cross-session checks pass; only GPU/toy checks fail."""
    manifest, docs, names = _zip_contents(resumed["zip"])
    blocks = manifest["blocks"]
    assert len(blocks) == _N_BLOCKS
    by_block = {block["block"]: block for block in blocks}
    assert by_block["humaneval_greedy"]["session"] == "20261025T000000Z-sessiona"
    assert by_block["humaneval_greedy"]["restored_in_session"] == "20261025T010000Z-sessionb"
    assert {block["session"] for name, block in by_block.items() if name != "humaneval_greedy"} == {
        "20261025T010000Z-sessionb"
    }
    assert [session["id"] for session in manifest["sessions"]] == [
        "20261025T000000Z-sessiona",
        "20261025T010000Z-sessionb",
    ]
    assert {"logs/20261025T000000Z-sessiona.log", "logs/20261025T010000Z-sessionb.log"} <= names
    assert manifest["session"] == "20261025T010000Z-sessionb"
    assert len(docs) == sum(len(block["files"]) for block in blocks)
    assert _check_blocks(manifest, docs, names) == []
    with pytest.raises(BundleError) as caught:
        validate_bundle(resumed["zip"])
    problems = {line.split(": ", 1)[-1] for line in str(caught.value).splitlines()[1:]}
    assert problems == {
        "measurement_kind 'toy_random_init_llama' != 'end_to_end'",
        "hardware.gpu is missing; this is not a GPU run",
    }, problems


def test_mixed_block_uses_references_from_the_restored_block(resumed: dict[str, Any]) -> None:
    """Mixed-block greedy matches cover HumanEval prompts whose references came from session A."""
    _manifest, docs, _names = _zip_contents(resumed["zip"])
    mixed = [doc for doc in docs.values() if doc["dataset"] == "mixed"]
    greedy = [doc for doc in mixed if doc["decoding"]["mode"] == "greedy"]
    assert greedy
    assert any(pid.startswith("humaneval/") for pid in greedy[0]["prompt_ids"])
    for doc in greedy:
        for key in ("greedy_token_match", "greedy_token_match_compiled"):
            match = doc[key]
            assert match["prompts_matched"] == match["n_prompts"] == 4, (doc["method"], match)
            assert match["source"] == "per-set blocks, by prompt id"


def test_bundle_without_a_kill_matches_the_resumed_one(
    resumed: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same result files, and the same greedy tokens, as one uninterrupted session."""
    monkeypatch.setattr(gpu_bundle, "_tracked_files_modified", lambda: False)
    zip_path, complete = gpu_bundle.build_bundle(
        [_SMOKE], tmp_path / "out", checkpoint_dir=tmp_path / "ckpt"
    )
    assert complete
    _m, fresh, _n = _zip_contents(zip_path)
    _m2, resumed_docs, _n2 = _zip_contents(resumed["zip"])
    assert set(fresh) == set(resumed_docs)
    for name, doc in fresh.items():
        if doc["decoding"]["mode"] == "greedy":
            assert doc["outputs"] == resumed_docs[name]["outputs"], name


class TestBlockCheckpoint:
    """What makes a saved block unusable."""

    def _saved(self, tmp_path: Path) -> tuple[BlockCheckpoint, Path]:
        result = tmp_path / "r.json"
        result.write_text(json.dumps({"git_commit": "abc", "x": 1}), encoding="utf-8")
        checkpoint = BlockCheckpoint(root=tmp_path / "ck", commit="abc", config_sha256="cfg")
        checkpoint.save("b_greedy", [result], started_utc="t", duration_s=1.0)
        return checkpoint, result

    def test_reused_when_intact(self, tmp_path: Path) -> None:
        """Same commit, config and bytes: restored byte for byte."""
        checkpoint, result = self._saved(tmp_path)
        meta = checkpoint.load("b_greedy")
        assert meta is not None and meta["files"] == {"r.json": sha256_file(result)}
        out = tmp_path / "out"
        (restored,) = checkpoint.restore("b_greedy", meta, out)
        assert restored.read_bytes() == result.read_bytes()

    def test_other_commit_config_or_changed_file_is_rerun(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Another commit, another config sha256, an edited file or no marker: run again."""
        checkpoint, _result = self._saved(tmp_path)
        root = checkpoint.root
        other_commit = BlockCheckpoint(root=root, commit="def", config_sha256="cfg")
        assert other_commit.load("b_greedy") is None
        other_config = BlockCheckpoint(root=root, commit="abc", config_sha256="new")
        assert other_config.load("b_greedy") is None
        saved = root / "blocks" / "b_greedy" / "r.json"
        saved.write_text(json.dumps({"git_commit": "abc", "x": 2}), encoding="utf-8")
        assert checkpoint.load("b_greedy") is None
        out = capsys.readouterr().out
        assert "saved for commit abc; running it again" in out
        assert "saved for a different config" in out
        assert "r.json is missing or changed" in out
        shutil.rmtree(root / "blocks" / "b_greedy")
        assert checkpoint.load("b_greedy") is None
