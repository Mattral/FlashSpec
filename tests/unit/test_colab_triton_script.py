"""The Colab Triton script writes a downloadable log and does not invent timings."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load() -> ModuleType:
    """Load the Colab script from its path.

    Returns
    -------
    module
        The script module.
    """
    path = Path("notebooks/colab_triton_tests.py")
    spec = importlib.util.spec_from_file_location("colab_triton_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_lines_report_cuda() -> None:
    """Record whether CUDA is present without requiring a GPU."""
    module = _load()
    blob = "\n".join(module.environment_lines())
    assert "cuda:" in blob
    assert "python:" in blob


def test_main_writes_the_download_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write the log from a stand-in pytest and name the file to download."""
    module = _load()
    monkeypatch.chdir(tmp_path)

    def _fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    code = module.main()
    assert code == 0
    text = (tmp_path / module.LOG_NAME).read_text(encoding="utf-8")
    assert "CI does not run these tests" in text
    assert "exit_code: 0" in text
    assert "1 passed" in text
    assert "Download this log" in text
