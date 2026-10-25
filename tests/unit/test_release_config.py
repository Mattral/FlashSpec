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


def test_030_config_differs_from_020_only_in_drafter_strategy() -> None:
    """The 0.3.0 release run is the 0.2.0 one with drafter_strategy: auto, nothing else."""
    previous = load_benchmark_config(_CONFIGS / "qwen_t4_compile_auto.yaml")
    release = load_benchmark_config(_CONFIGS / "qwen_t4_release_030.yaml")
    assert release.pop("drafter_strategy") == "auto"
    assert "drafter_strategy" not in previous
    assert release == previous
    assert {"flashspec_compiled", "hf_greedy_compiled", "hf_sample_compiled"} <= set(
        release["methods"]
    )
    assert release["datasets"] == ["mt_bench", "humaneval"]
    assert release["temperatures"] == [0.0, 0.7]


def test_harness_drafter_strategy_key() -> None:
    """Absent means the pre-0.3.0 model-drafter path; unknown values are refused."""
    import pytest

    from benchmarks.harness import _drafter_strategy

    assert _drafter_strategy({}) == "bandit"
    assert _drafter_strategy({"drafter_strategy": "auto"}) == "auto"
    with pytest.raises(ValueError, match="drafter_strategy"):
        _drafter_strategy({"drafter_strategy": "composite-g5"})
