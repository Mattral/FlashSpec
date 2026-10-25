"""``benchmarks/phase6.py`` end to end on a random tiny Llama (CPU), plus report and ingest checks.

A correctness run of the pipeline only: every number is from random
weights on CPU and means nothing as a speed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import load_benchmark_config, run_config_file
from benchmarks.phase6 import run_phase6_config
from scripts.ingest_gpu_bundle import _check_phase6, _readme_documents
from scripts.phase6_report import evaluate, load_documents, render

_ARMS = [
    {"label": "qwen-g2", "drafter": "model", "gamma": 2},
    {"label": "ngram-g4", "drafter": "ngram", "gamma": 4},
    {"label": "composite-g4", "drafter": "ngram", "gamma": 4, "fallback": "model"},
]


def _config(temperatures: list[float], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runner": "phase6",
        "model": "toy_llama",
        "drafter": "toy_llama",
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": 8,
        "n_warmup": 1,
        "seed": 0,
        "compile": "auto",
        "vocab_size": 64,
        "target_hidden": 32,
        "target_layers": 2,
        "drafter_hidden": 32,
        "drafter_layers": 1,
        "temperatures": temperatures,
        "arms": _ARMS,
        "bandits": ["ucb1", "thompson"],
        "blocks": blocks,
    }


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


@pytest.fixture(scope="module")
def greedy_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Two per-set blocks (HF references) and a short mixed block with every FlashSpec method."""
    out = tmp_path_factory.mktemp("phase6")
    hf = ["hf_greedy", "hf_greedy_compiled", "arm_qwen-g2"]
    config = _config(
        [0.0],
        [
            {"dataset": "humaneval", "methods": hf},
            {"dataset": "mt_bench", "methods": hf},
            {
                "dataset": "mixed",
                "n_prompts": 3,
                "methods": [
                    "fixed_arms",
                    "bandits",
                    "flashspec_compiled_legacy",
                    "bandit_ucb1_legacy",
                ],
            },
        ],
    )
    written = run_phase6_config(out / "toy_phase6.yaml", out, config=config)
    assert len(written) == 3 + 3 + 7
    return load_documents(out)


def test_documents_and_mixed_references(greedy_run: list[dict[str, Any]]) -> None:
    """Mixed-block methods are matched against the per-set HF outputs by prompt id."""
    mixed = {doc["method"]: doc for doc in greedy_run if doc["dataset"] == "mixed"}
    assert set(mixed) == {
        "arm_qwen-g2",
        "arm_ngram-g4",
        "arm_composite-g4",
        "bandit_ucb1",
        "bandit_thompson",
        "flashspec_compiled_legacy",
        "bandit_ucb1_legacy",
    }
    for doc in mixed.values():
        for key in ("greedy_token_match", "greedy_token_match_compiled"):
            match = doc[key]
            assert match["source"] == "per-set blocks, by prompt id"
            assert match["prompts_matched"] == match["n_prompts"] == 3, (doc["method"], match)
        assert _check_phase6(doc["method"], doc) == []
        assert doc["prompt_ids"][0] == "mt_bench/121"
        assert doc["outputs"]["prompts"][0]["prompt_id"] == "mt_bench/121"
    bandit = mixed["bandit_ucb1"]
    report = bandit["arm_report"]
    assert [arm["label"] for arm in report["per_arm"]] == ["qwen-g2", "ngram-g4", "composite-g4"]
    assert len(report["selection_over_time"]["per_prompt"]) == 3
    assert sum(arm["pulls"] for arm in report["per_arm"]) == report["n_steps"]
    assert bandit["bandit"]["selector"] == "ucb1"
    assert bandit["compile"]["fixed_shapes"] is True
    assert mixed["bandit_ucb1_legacy"]["compile"]["fixed_shapes"] is False
    assert mixed["bandit_ucb1"]["cuda_graphs"]["group"] == "verify_width_5"
    assert mixed["arm_qwen-g2"]["cuda_graphs"]["group"] == "verify_width_3"
    assert mixed["flashspec_compiled_legacy"]["cuda_graphs"]["group"].startswith("legacy_")
    assert bandit["host_syncs"]["by_kind"] == {"tolist": bandit["host_syncs"]["total"]}
    metrics = bandit["metrics"]
    assert metrics["workload_tokens_per_second"] > 0
    assert "workload_speedup_vs_arm_ngram-g4" in metrics
    assert "speedup_vs_ar_mean" in bandit["unmeasured"]


def test_report_and_readme_filter(greedy_run: list[dict[str, Any]]) -> None:
    """The report evaluates every criterion it has inputs for; README tables skip phase6."""
    result = evaluate(greedy_run)
    greedy = result["greedy"]
    assert set(greedy["blocks"]) == {"humaneval", "mt_bench", "mixed"}
    mixed = greedy["blocks"]["mixed"]
    assert mixed["best_fixed_arm"] in {"arm_qwen-g2", "arm_ngram-g4", "arm_composite-g4"}
    best = mixed["best_fixed_tps"]
    assert mixed["bandit_ucb1"]["vs_best_fixed"] == mixed["bandit_ucb1"]["tps"] / best
    assert greedy["criteria"]["1_mixed_beats_best_fixed"] in ("pass", "FAIL")
    # humaneval_edit was not run here: the per-set criteria cannot pass.
    assert greedy["criteria"]["2_each_set_within_5pct_of_best_fixed"] == "n/a"
    assert greedy["criteria"]["4_greedy_matches_compiled_hf"] == "pass"
    assert {"verify_width_3", "verify_width_5"} <= set(result["cuda_graphs"])
    assert "## greedy" in render(result)
    assert _readme_documents(greedy_run, None) == []


def test_sampled_block_and_config_dispatch(tmp_path: Path) -> None:
    """T = 0.7: no greedy matches, hf_sample_compiled runs; run_config_file dispatches."""
    config_path = tmp_path / "toy_phase6_sample.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runner: phase6",
                "model: toy_llama",
                "drafter: toy_llama",
                "device: cpu",
                "dtype: float32",
                "max_new_tokens: 6",
                "n_warmup: 0",
                "vocab_size: 64",
                "target_hidden: 32",
                "target_layers: 1",
                "drafter_hidden: 32",
                "drafter_layers: 1",
                "temperatures: [0.7]",
                "arms:",
                "  - {label: qwen-g2, drafter: model, gamma: 2}",
                "  - {label: composite-g4, drafter: ngram, gamma: 4, fallback: model}",
                "bandits: [thompson]",
                "blocks:",
                "  - dataset: humaneval_edit",
                "    n_prompts: 2",
                "    methods: [fixed_arms, bandits, hf_greedy, hf_sample_compiled]",
            ]
        ),
        encoding="utf-8",
    )
    assert load_benchmark_config(config_path)["runner"] == "phase6"
    written = run_config_file(config_path, tmp_path / "out")
    names = sorted(path.name for path in written)
    assert len(names) == 4  # two arms, one bandit, hf_sample_compiled (hf_greedy skipped)
    docs = load_documents(tmp_path / "out")
    for doc in docs:
        assert "greedy_token_match" not in doc
        assert _check_phase6(doc["method"], doc) == []
    thompson = next(doc for doc in docs if doc["method"] == "bandit_thompson")
    assert thompson["bandit"]["seed"] == 0


def test_bad_configs_raise(tmp_path: Path) -> None:
    """Unknown methods, missing arms and synthetic datasets are refused."""
    bad = _config([0.0], [{"dataset": "humaneval", "methods": ["nope"]}])
    with pytest.raises(RuntimeError, match="Unknown phase6 method"):
        run_phase6_config(tmp_path / "x.yaml", tmp_path, config=bad)
    with pytest.raises(RuntimeError, match="arms"):
        run_phase6_config(tmp_path / "x.yaml", tmp_path, config={**bad, "arms": []})
    synthetic = _config([0.0], [{"dataset": "synthetic_prompts", "methods": ["fixed_arms"]}])
    with pytest.raises(RuntimeError, match="fixed prompt sets"):
        run_phase6_config(tmp_path / "x.yaml", tmp_path, config=synthetic)
