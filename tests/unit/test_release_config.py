"""The 0.2.0 release config is the Phase 5 Qwen config plus ``flashspec_compile: auto``."""

from __future__ import annotations

from pathlib import Path

from benchmarks.harness import load_benchmark_config

_CONFIGS = Path("benchmarks/configs")


def test_auto_config_differs_from_phase5_only_in_compile_request() -> None:
    """Same pair, prompts, modes, lengths and methods; only flashspec_compile is added."""
    base = load_benchmark_config(_CONFIGS / "qwen_t4_compile.yaml")
    auto = load_benchmark_config(_CONFIGS / "qwen_t4_compile_auto.yaml")
    assert auto.pop("flashspec_compile") == "auto"
    assert "flashspec_compile" not in base
    assert auto == base
    assert "flashspec_compiled" in auto["methods"]
    assert "hf_greedy_compiled" in auto["methods"]


def test_package_version_matches_pyproject() -> None:
    """publish.yml checks the tag against pyproject; the package must report the same."""
    import tomllib

    import flashspec

    with open("pyproject.toml", "rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    assert flashspec.__version__ == version
