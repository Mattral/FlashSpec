"""The composite comparison run: tie check, runner wiring, report, and the pre-registered config.

Toy models on CPU: correctness of the pipeline only; no number here is a speed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from benchmarks.harness import load_benchmark_config
from benchmarks.phase6 import (
    _engine,
    _Group,
    _Method,
    _parse_arms,
    _tie_check,
    run_phase6_config,
)
from benchmarks.tie_check import capture_verify_rows, classify, first_divergence
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import ArmSpec, strategy_arms
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.utils.config import FlashSpecConfig, SamplingConfig
from scripts.composite_report import evaluate, load_documents, render
from scripts.ingest_gpu_bundle import _check_phase6

_CONFIGS = Path(__file__).resolve().parents[2] / "benchmarks" / "configs"
_ARMS = [
    {"label": "composite-g4", "drafter": "ngram", "gamma": 4, "fallback": "model"},
    {
        "label": "composite-g6-f2",
        "drafter": "ngram",
        "gamma": 6,
        "fallback": "model",
        "fallback_gamma": 2,
    },
]


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


# ── classify ─────────────────────────────────────────────────────────────────


def _row(values: dict[int, float], vocab: int = 8) -> torch.Tensor:
    row = torch.full((vocab,), -5.0)
    for index, value in values.items():
        row[index] = value
    return row


def _classify(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    reference = [1, 2, 3, 4]
    path = [1, 2, 5, 6]
    kwargs: dict[str, Any] = {
        "dtype": "float16",
        "reference": reference,
        "path": path,
        "reference_replay": list(reference),
        "path_replay": list(path),
        # Reference: tokens 3 and 5 tie exactly; path: 5 wins by 0.015625.
        "reference_row": _row({3: 2.0, 5: 2.0}),
        "path_row": _row({3: 2.0, 5: 2.015625}),
    }
    kwargs.update(overrides)
    return classify(**kwargs)


class TestClassify:
    """The allowed-exact-tie rule."""

    def test_exact_tie_is_allowed(self) -> None:
        """Same prefix, same top-2 pair, one margin exactly 0.0, fp16: allowed."""
        entry = _classify()
        assert entry["position"] == 2
        assert (entry["reference_token"], entry["path_token"]) == (3, 5)
        assert entry["reference_top2"]["margin_top1_top2"] == 0.0
        assert entry["path_top2"]["margin_top1_top2"] == 0.015625
        assert entry["one_margin_exactly_zero"] is True
        assert entry["diverging_tokens_are_both_rows_top2"] is True
        assert entry["allowed_exact_tie"] is True

    def test_each_condition_bites(self) -> None:
        """fp32, a non-zero margin on both sides, a third token, or a bad replay: not allowed."""
        assert _classify(dtype="float32")["allowed_exact_tie"] is False
        near = _classify(reference_row=_row({3: 2.0078125, 5: 2.0}))
        assert near["one_margin_exactly_zero"] is False
        assert near["allowed_exact_tie"] is False
        other = _classify(path_row=_row({3: 2.0, 7: 2.0}))
        assert other["diverging_tokens_are_both_rows_top2"] is False
        assert other["allowed_exact_tie"] is False
        replay = _classify(reference_replay=[1, 2, 3, 0])
        assert replay["replays_reproduce_recorded"] is False
        assert replay["allowed_exact_tie"] is False
        missing = _classify(path_row=None)
        assert missing["allowed_exact_tie"] is False

    def test_no_divergence(self) -> None:
        """Equal outputs have nothing to classify."""
        entry = _classify(path=[1, 2, 3, 4], path_replay=[1, 2, 3, 4])
        assert entry["position"] is None
        assert entry["allowed_exact_tie"] is None
        assert first_divergence([1, 2], [1, 2, 3]) == 2


# ── capture_verify_rows ──────────────────────────────────────────────────────


def _toy_llama(layers: int, seed: int) -> Any:  # noqa: ANN401
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        initializer_range=0.1,
    )
    return LlamaForCausalLM(config).eval()


@pytest.mark.parametrize("static", [False, True])
def test_captured_rows_chose_every_greedy_token(static: bool) -> None:
    """Row ``k`` is the verify row that chose token ``k``: its argmax is that token."""
    target = TargetModel("t", torch.device("cpu"), torch.float32, model=_toy_llama(2, 3))
    drafter = HuggingFaceDrafter("d", device="cpu", model=_toy_llama(1, 4))
    if static:
        target.use_static_cache(128, compile_forward=False, pad_widths=(7,))
    lookup = [False, True]
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=24,
        drafter_strategy="composite-g6-f2",
        sampling=SamplingConfig(temperature=0.0),
    )
    engine = SpeculativeEngine(
        config,
        [drafter, NgramDrafter()],  # type: ignore[list-item]
        target,
        UCB1Selector(n_arms=1, window_size=0),
        arms=strategy_arms("composite-g6-f2", lookup, 4),
    )
    chunk = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(5))
    prompt = torch.cat([chunk, chunk], dim=1)
    with capture_verify_rows(target, int(prompt.shape[1])) as rows:
        out = engine.generate(prompt).output_ids[0].tolist()
    assert "_verify_rows" not in vars(target)  # the wrapper is removed
    assert set(range(len(out))) <= set(rows)
    assert [int(rows[k].argmax()) for k in range(len(out))] == out


# ── the runner with greedy_tie_check, then the report ────────────────────────


def _config(blocks: list[dict[str, Any]], temperatures: list[float]) -> dict[str, Any]:
    return {
        "runner": "phase6",
        "model": "toy_llama",
        "drafter": "toy_llama",
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": 8,
        "n_warmup": 0,
        "seed": 0,
        "compile": "auto",
        "vocab_size": 64,
        "target_hidden": 32,
        "target_layers": 2,
        "drafter_hidden": 32,
        "drafter_layers": 1,
        "greedy_tie_check": True,
        "temperatures": temperatures,
        "arms": _ARMS,
        "bandits": [],
        "blocks": blocks,
    }


@pytest.fixture(scope="module")
def composite_run(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, Any]]:
    """Run the shape of qwen_t4_composite.yaml on a toy pair; mixed is capped at 3 prompts.

    The per-set blocks run every prompt, so the mixed block finds its
    greedy references by prompt id.
    """
    out = tmp_path_factory.mktemp("composite")
    per_set = ["fixed_arms", "hf_greedy", "hf_greedy_compiled", "hf_sample_compiled"]
    blocks: list[dict[str, Any]] = [
        {"dataset": dataset, "methods": per_set}
        for dataset in ("humaneval", "mt_bench", "humaneval_edit")
    ]
    blocks.append({"dataset": "mixed", "n_prompts": 3, "methods": ["fixed_arms"]})
    written = run_phase6_config(out / "toy_composite.yaml", out, config=_config(blocks, [0.0, 0.7]))
    # Per set: 2 arms + hf_greedy + one compiled HF per mode; mixed: 2 arms per mode.
    assert len(written) == 3 * (4 + 3) + 2 * 2
    return load_documents(out)


def test_runner_records_arms_and_tie_check(composite_run: list[dict[str, Any]]) -> None:
    """fallback_gamma reaches the documents; fp32 toy greedy has nothing to classify."""
    flashspec = [doc for doc in composite_run if "arms" in doc]
    assert {doc["method"] for doc in flashspec} == {"arm_composite-g4", "arm_composite-g6-f2"}
    for doc in flashspec:
        assert _check_phase6(doc["method"], doc) == []
        (arm,) = doc["arms"]
        split = doc["method"] == "arm_composite-g6-f2"
        assert (arm["gamma"], arm["fallback_gamma"]) == ((6, 2) if split else (4, 4))
        assert doc["gamma"] == (6 if split else 4)
        assert doc["cuda_graphs"]["group"] == ("verify_width_7" if split else "verify_width_5")
        assert doc["arm_report"]["per_arm"][0]["fallback_gamma"] == (2 if split else 4)
        if doc["decoding"]["mode"] == "greedy":
            check = doc["greedy_tie_check"]
            assert check["checked"] is True
            assert check["n_divergent"] == 0
            assert check["all_divergences_allowed"] is True
            match = doc["greedy_token_match_compiled"]
            assert match["prompts_matched"] == match["n_prompts"]
        else:
            assert "greedy_tie_check" not in doc


def test_report_applies_the_rule(composite_run: list[dict[str, Any]]) -> None:
    """Every input is present, so each check is pass or FAIL, never n/a."""
    result = evaluate(composite_run)
    decision = result["decision"]
    assert decision["pick"] in ("arm_composite-g4", "arm_composite-g6-f2")
    mixed = decision["mixed_greedy_tps"]
    assert mixed[decision["pick"]] == max(mixed.values())
    pick = result["variants"][decision["pick"]]
    assert pick["check_3_greedy_matches_compiled_hf_or_exact_tie"] == "pass"
    assert pick["check_2_beats_compiled_hf_every_set_both_modes"] in ("pass", "FAIL")
    assert decision["verdict"] in ("pass", "FAIL")
    assert (decision["default"] is not None) == (decision["verdict"] == "pass")
    assert len(pick["vs_hf"]) == 6
    assert "Pick (higher mixed greedy throughput)" in render(result)


def test_tie_check_replays_a_divergence(tmp_path: Path) -> None:
    """A recorded reference that the HF replay does not reproduce is not an allowed tie."""
    from transformers import StaticCache

    target_module, drafter_module = _toy_llama(2, 3), _toy_llama(1, 4)
    group = _Group(
        "verify_width_7",
        TargetModel("t", torch.device("cpu"), torch.float32, model=target_module),
        HuggingFaceDrafter("d", device="cpu", model=drafter_module),
        NgramDrafter(),
    )
    method = _Method("arm_composite-g6-f2", _parse_arms({"arms": _ARMS})[1:], "fixed", True, group)

    def make(name: str) -> SpeculativeEngine:
        return _engine(
            method,
            temperature=0.0,
            seed=0,
            window=0,
            compile_request="auto",
            cache_len=96,
            max_new_tokens=12,
            device=torch.device("cpu"),
            weight_dtype=torch.float32,
        )

    prompt = torch.randint(0, 128, (1, 10), generator=torch.Generator().manual_seed(8))
    got = make(method.name).generate(prompt).output_ids[0].tolist()
    corrupted = list(got)
    corrupted[5] = (corrupted[5] + 1) % 128
    check = _tie_check(
        method.name,
        method=method,
        make=make,
        outputs=[got],
        prompt_ids=["p0"],
        prompts=[prompt],
        wanted=[corrupted],
        target_module=target_module,
        hf_cache=StaticCache(config=target_module.config, max_cache_len=96),
        hf_replays={},
        max_new_tokens=12,
    )
    assert check["checked"] is True and check["n_divergent"] == 1
    (entry,) = check["divergences"]  # type: ignore[misc]
    assert entry["prompt_id"] == "p0" and entry["position"] == 5
    assert entry["replays_reproduce_recorded"] is False
    assert entry["allowed_exact_tie"] is False
    assert check["all_divergences_allowed"] is False
    no_ref = _tie_check(
        method.name,
        method=method,
        make=make,
        outputs=[got],
        prompt_ids=["p0"],
        prompts=[prompt],
        wanted=[None],
        target_module=target_module,
        hf_cache=None,
        hf_replays={},
        max_new_tokens=12,
    )
    assert no_ref["checked"] is False


# ── report on hand-made documents ────────────────────────────────────────────


def _doc(
    method: str,
    dataset: str,
    mode: str,
    tps: float,
    **extra: Any,  # noqa: ANN401
) -> dict[str, Any]:
    decoding = {"mode": "greedy"} if mode == "greedy" else {"mode": "sample", "temperature": 0.7}
    return {
        "phase6": {},
        "git_commit": "c0ffee0",
        "method": method,
        "dataset": dataset,
        "decoding": decoding,
        "metrics": {"workload_tokens_per_second": tps},
        **extra,
    }


def _docs(
    g4_mixed: float, g6_mixed: float, hf: float, g6_greedy_extra: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    exact = {"greedy_token_match_compiled": {"prompts_matched": 10, "n_prompts": 10}}
    docs: list[dict[str, Any]] = []
    for mode, hf_name in (("greedy", "hf_greedy_compiled"), ("t0.7", "hf_sample_compiled")):
        for dataset in ("humaneval", "mt_bench", "humaneval_edit", "mixed"):
            extra_g4 = exact if mode == "greedy" else {}
            extra_g6 = (g6_greedy_extra or exact) if mode == "greedy" else {}
            g4 = g4_mixed if dataset == "mixed" else 100.0
            g6 = g6_mixed if dataset == "mixed" else 100.0
            docs.append(_doc("arm_composite-g4", dataset, mode, g4, **extra_g4))
            docs.append(_doc("arm_composite-g6-f2", dataset, mode, g6, **extra_g6))
            if dataset != "mixed":
                docs.append(_doc(hf_name, dataset, mode, hf))
    return docs


def test_report_rule_on_made_up_documents() -> None:
    """Higher mixed greedy wins; it must beat HF everywhere; ties pass, other divergences fail."""
    result = evaluate(_docs(80.0, 90.0, 50.0))
    assert result["decision"]["pick"] == "arm_composite-g6-f2"
    assert result["decision"]["verdict"] == "pass"
    assert result["decision"]["default"] == "composite-g6-f2"
    assert result["decision"]["pick_over_other_mixed_greedy"] == {"arm_composite-g4": 90.0 / 80.0}

    # The pick is slower than HF somewhere: no default, and the other variant is not used.
    slow = evaluate(_docs(80.0, 90.0, 150.0))
    assert slow["decision"]["pick"] == "arm_composite-g6-f2"
    assert slow["decision"]["verdict"] == "FAIL"
    assert slow["decision"]["default"] is None

    def tie_doc(allowed: bool) -> dict[str, Any]:
        return {
            "greedy_token_match_compiled": {
                "prompts_matched": 9,
                "n_prompts": 10,
                "first_divergence": [None] * 9 + [74],
            },
            "greedy_tie_check": {
                "checked": True,
                "divergences": [{"prompt_id": "81", "position": 74, "allowed_exact_tie": allowed}],
            },
        }

    tie = evaluate(_docs(80.0, 90.0, 50.0, tie_doc(True)))
    assert tie["decision"]["verdict"] == "pass"
    assert tie["variants"]["arm_composite-g6-f2"]["greedy"]["mixed"]["exact_ties"] == 1
    bad = evaluate(_docs(80.0, 90.0, 50.0, tie_doc(False)))
    assert bad["decision"]["verdict"] == "FAIL"
    assert bad["variants"]["arm_composite-g6-f2"]["greedy"]["mixed"]["not_allowed"] == [
        {"prompt_id": "81", "position": 74}
    ]
    unchecked = evaluate(
        _docs(
            80.0,
            90.0,
            50.0,
            {"greedy_token_match_compiled": {"prompts_matched": 9, "n_prompts": 10}},
        )
    )
    assert unchecked["decision"]["verdict"] == "n/a"
    missing = evaluate([d for d in _docs(80.0, 90.0, 50.0) if d["dataset"] != "mixed"])
    assert missing["decision"]["verdict"] == "n/a" and missing["decision"]["pick"] is None


# ── the pre-registered config ────────────────────────────────────────────────


def test_config_matches_the_phase6_setup() -> None:
    """Same models, dtype, prompts, modes and seeds as Phase 6; the two variants only."""
    composite = load_benchmark_config(_CONFIGS / "qwen_t4_composite.yaml")
    phase6 = load_benchmark_config(_CONFIGS / "qwen_t4_phase6.yaml")
    for key in (
        "runner",
        "model",
        "drafter",
        "device",
        "dtype",
        "max_new_tokens",
        "n_warmup",
        "seed",
        "chat_template",
        "compile",
        "compile_max_cache_len",
        "temperatures",
    ):
        assert composite[key] == phase6[key], key
    assert [b["dataset"] for b in composite["blocks"]] == [b["dataset"] for b in phase6["blocks"]]
    assert not any("n_prompts" in block for block in composite["blocks"])
    assert composite["greedy_tie_check"] is True
    assert composite["bandits"] == []
    arms = _parse_arms(composite)
    assert arms == [
        ArmSpec(1, 4, "composite-g4", fallback=0),
        ArmSpec(1, 6, "composite-g6-f2", fallback=0, fallback_gamma=2),
    ]
    # composite-g4 is the Phase 6 composite arm, unchanged.
    assert next(a for a in _parse_arms(phase6) if a.label == "composite-g4") == arms[0]
    # Each variant equals the arm its drafter_strategy builds (lookup drafter 1, model 0).
    for arm in arms:
        (built,) = strategy_arms(arm.label, [False, True], 4)
        assert built == arm
    for block in composite["blocks"][:3]:
        assert {"hf_greedy_compiled", "hf_sample_compiled"} <= set(block["methods"])
