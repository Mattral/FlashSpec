"""The harness under ``drafter_strategy: auto`` (the 0.3.0 release run), on a toy pair (CPU).

A correctness run of the pipeline only; no number here is a speed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from benchmarks.harness import run_config_file
from scripts.ingest_gpu_bundle import _drafting, render_status, render_tables

_TOY = """\
model: toy_llama
drafter: toy_llama
device: cpu
dtype: float32
gamma: 4
batch_size: 1
max_new_tokens: 32
n_warmup: 1
n_samples: 3
prompt_length: 16
vocab_size: 64
target_hidden: 32
target_layers: 2
drafter_hidden: 32
drafter_layers: 1
initializer_range: 0.1
seed: 0
dataset: synthetic_prompts
flashspec_compile: auto
temperatures: [0.0, 0.7]
methods: [flashspec_ucb, flashspec_compiled, hf_greedy, hf_greedy_compiled, hf_sample_compiled]
"""


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _run(tmp: Path, strategy: str | None) -> list[dict[str, Any]]:
    config = tmp / "toy.yaml"
    extra = f"drafter_strategy: {strategy}\n" if strategy else ""
    config.write_text(_TOY + extra, encoding="utf-8")
    files = run_config_file(config, tmp / "out")
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


@pytest.fixture(scope="module")
def auto_docs(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Run the toy config once with drafter_strategy: auto."""
    return _run(tmp_path_factory.mktemp("auto"), "auto")


def test_documents_record_the_resolved_strategy(auto_docs: list[dict[str, Any]]) -> None:
    """Both FlashSpec methods resolve auto to composite-g6-f2 and say so."""
    spec = [doc for doc in auto_docs if "drafter_strategy" in doc]
    assert {doc["method"] for doc in spec} == {"flashspec_ucb", "flashspec_compiled"}
    for doc in spec:
        strategy = doc["drafter_strategy"]
        assert (strategy["requested"], strategy["resolved"]) == ("auto", "composite-g6-f2")
        assert strategy["arms"] == [
            {"label": "composite-g6-f2", "gamma": 6, "fallback_gamma": 2, "lookup": True}
        ]
        report = doc["arm_report"]["per_arm"][0]
        split = report["steps_by_drafter"]
        assert split["lookup"] + split["fallback"] == report["pulls"] > 0


def test_derived_drafter_forwards_equal_the_hooked_count(
    auto_docs: list[dict[str, Any]],
) -> None:
    """The compiled method's drafter forwards, derived from arm records, match the eager hook.

    On CPU both methods take the same steps, so the eager engine's hooked
    count is the ground truth for the derivation.
    """
    by_key = {(d["method"], d["decoding"]["mode"]): d for d in auto_docs if "arm_report" in d}
    for mode in ("greedy", "sample"):
        eager = by_key[("flashspec_ucb", mode)]["forward_counts"]
        compiled = by_key[("flashspec_compiled", mode)]["forward_counts"]
        assert eager["source"] == "measured with a forward hook"
        assert "arm records" in compiled["source"]
        assert compiled["drafter_forwards"] == eager["drafter_forwards"]


def test_greedy_matches_and_ingest_text(auto_docs: list[dict[str, Any]]) -> None:
    """Greedy auto output equals HF; the README text names the strategy, not gamma 4."""
    compiled = next(
        d
        for d in auto_docs
        if d["method"] == "flashspec_compiled" and d["decoding"]["mode"] == "greedy"
    )
    match = compiled["greedy_token_match_compiled"]
    assert match["prompts_matched"] == match["n_prompts"]
    text = _drafting(compiled)
    assert "composite-g6-f2" in text and "requested `auto`" in text
    status = render_status(auto_docs)
    assert status is not None and "composite-g6-f2" in status and "gamma 4" not in status
    assert "composite-g6-f2" in render_tables(auto_docs)


def test_default_key_keeps_the_model_drafter_path(tmp_path: Path) -> None:
    """Without the key the harness measures what it did before 0.3.0, text unchanged."""
    docs = _run(tmp_path, None)
    for doc in (d for d in docs if "drafter_strategy" in d):
        assert doc["drafter_strategy"]["resolved"] == "bandit"
        assert doc["drafter_strategy"]["arms"][0]["lookup"] is False
        assert _drafting(doc) == "gamma 4"
