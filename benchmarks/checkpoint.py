"""Per-block checkpoints for long runs, so a lost session loses at most one block.

A *block* is one (prompt set, decoding mode) unit of a ``runner: phase6``
config. When a block finishes, :meth:`BlockCheckpoint.save` copies its result
JSONs under ``<root>/blocks/<key>/``, then writes ``block.json`` (the marker,
written last and atomically) with the commit, the config's sha256, the
session id, a sha256 per file and whether tracked files were modified. It
also copies this session's log so far to ``<root>/logs/<session>.log``.

On a rerun, :meth:`BlockCheckpoint.load` accepts a block only when its
marker names the same commit and config sha256 and every file still has
its recorded sha256 and commit; anything else is run again, with the
reason printed. In the notebook ``root`` is a Google Drive folder per
commit, which survives a Colab disconnect. Drive syncs in the background,
so a machine lost seconds after a save may leave a block half synced;
the hash checks then reject it, and the worst case is running that one
block again.

Nothing here reads or writes a measured number: files are copied byte for
byte and checked by hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["SESSION_ID", "BlockCheckpoint", "sha256_file"]

#: This process's session id: start time (UTC) and a random suffix. No host details.
SESSION_ID = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"

_MARKER = "block.json"


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class BlockCheckpoint:
    """Save and restore finished blocks under ``root``.

    Parameters
    ----------
    root : Path
        Checkpoint folder for one commit (created if missing).
    commit : str
        The commit being measured; blocks from another commit are not reused.
    config_sha256 : str
        sha256 of the config file; blocks from another config are not reused.
    log_path : Path or None
        This session's log, copied to ``logs/<session>.log`` after each block.
    session : str
        This session's id (default :data:`SESSION_ID`).
    tracked_files_modified : callable
        Returns whether tracked files differ from the commit, recorded per block.
    """

    root: Path
    commit: str
    config_sha256: str
    log_path: Path | None = None
    session: str = SESSION_ID
    tracked_files_modified: Callable[[], bool | None] = lambda: None
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        (self.root / "blocks").mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

    def _dir(self, key: str) -> Path:
        return self.root / "blocks" / key

    def load(self, key: str) -> dict[str, Any] | None:
        """Return the marker of a finished, intact block ``key``, or ``None`` (run it)."""
        marker = self._dir(key) / _MARKER
        if not marker.is_file():
            return None
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[checkpoint] {key}: unreadable marker ({exc}); running it again")
            return None
        if meta.get("git_commit") != self.commit:
            print(
                f"[checkpoint] {key}: saved for commit {meta.get('git_commit')}; running it again"
            )
            return None
        if meta.get("config_sha256") != self.config_sha256:
            print(f"[checkpoint] {key}: saved for a different config; running it again")
            return None
        for name, digest in (meta.get("files") or {}).items():
            path = self._dir(key) / name
            if not path.is_file() or sha256_file(path) != digest:
                print(f"[checkpoint] {key}: {name} is missing or changed; running it again")
                return None
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                print(f"[checkpoint] {key}: {name} is not JSON; running it again")
                return None
            if doc.get("git_commit") != self.commit:
                print(f"[checkpoint] {key}: {name} names another commit; running it again")
                return None
        if not meta.get("files"):
            return None
        return meta  # type: ignore[no-any-return]

    def restore(self, key: str, meta: dict[str, Any], results_dir: Path) -> list[Path]:
        """Copy a loaded block's files into ``results_dir`` and record it as restored."""
        results_dir.mkdir(parents=True, exist_ok=True)
        restored = []
        for name in meta["files"]:
            destination = results_dir / name
            shutil.copyfile(self._dir(key) / name, destination)
            restored.append(destination)
        self.blocks.append({**meta, "restored_in_session": self.session})
        return restored

    def save(
        self,
        key: str,
        files: list[Path],
        *,
        started_utc: str,
        duration_s: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Copy ``files`` for block ``key``, then write its marker; return the marker."""
        directory = self._dir(key)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        digests: dict[str, str] = {}
        for path in files:
            shutil.copyfile(path, directory / path.name)
            digests[path.name] = sha256_file(directory / path.name)
        meta: dict[str, Any] = {
            "block": key,
            "git_commit": self.commit,
            "config_sha256": self.config_sha256,
            "session": self.session,
            "started_utc": started_utc,
            "finished_utc": datetime.now(UTC).isoformat(),
            "duration_s": duration_s,
            "tracked_files_modified": self.tracked_files_modified(),
            "files": digests,
            **(extra or {}),
        }
        # The marker goes last: a block without one was not finished.
        _atomic_write(directory / _MARKER, json.dumps(meta, indent=2) + "\n")
        self.save_log()
        self.blocks.append(meta)
        return meta

    def save_log(self) -> None:
        """Copy this session's log so far to ``logs/<session>.log``."""
        if self.log_path is not None and self.log_path.is_file():
            destination = self.root / "logs" / f"{self.session}.log"
            tmp = destination.with_name(f".{destination.name}.tmp")
            shutil.copyfile(self.log_path, tmp)
            os.replace(tmp, destination)

    def session_logs(self) -> dict[str, Path]:
        """Every saved session log, by session id."""
        return {path.stem: path for path in sorted((self.root / "logs").glob("*.log"))}
