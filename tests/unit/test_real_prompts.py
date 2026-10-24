"""Real prompt sets are tokenized text, not synthetic ids labeled as a dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks.harness import (
    DeviceUnavailableError,
    _prompts,
    _result_filename,
    assemble_result,
    run_config_file,
    run_paired_generations,
)
from benchmarks.prompt_sets import load_prompt_set, tokenize_prompts
from flashspec.engine.speculative import SpeculativeEngine


class _CharTokenizer:
    """Map each character to a token id inside a 64-word vocabulary."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode ``text`` one character at a time.

        Parameters
        ----------
        text : str
            Prompt text.
        add_special_tokens : bool
            Accepted so the harness can pass the HuggingFace flag.

        Returns
        -------
        list[int]
            Ids in ``1..50``.
        """
        assert add_special_tokens
        return [(ord(character) % 50) + 1 for character in text]


def _toy_config(dataset: str) -> dict[str, object]:
    """Return a tiny toy config that can score one short prompt.

    Parameters
    ----------
    dataset : str
        Dataset name passed to the harness.

    Returns
    -------
    dict[str, object]
        Config with an injected tokenizer.
    """
    return {
        "model": "toy",
        "device": "cpu",
        "dataset": dataset,
        "tokenizer": _CharTokenizer(),
        "gamma": 2,
        "max_new_tokens": 2,
        "n_warmup": 0,
        "n_samples": 1,
        "prompt_length": 8,
        "vocab_size": 64,
        "target_hidden": 32,
        "target_layers": 1,
        "drafter_hidden": 32,
        "drafter_layers": 1,
        "seed": 0,
    }


def test_mt_bench_without_a_tokenizer_writes_nothing(tmp_path: Path) -> None:
    """Refuse a toy MT-Bench label that has no tokenizer."""
    config = tmp_path / "mt_bench.yaml"
    config.write_text(
        "\n".join(
            [
                "model: toy",
                "device: cpu",
                "dataset: mt_bench",
                "gamma: 2",
                "max_new_tokens: 2",
                "n_warmup: 0",
                "n_samples: 1",
                "prompt_length: 8",
                "vocab_size: 32",
                "target_hidden: 32",
                "target_layers: 1",
                "drafter_hidden: 32",
                "drafter_layers: 1",
                "seed: 0",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="tokenizer"):
        run_config_file(config, output)
    assert list(output.glob("*.json")) == []


def test_fixed_sets_keep_the_selected_ids() -> None:
    """Load the vendored ids and refuse Alpaca."""
    humaneval = load_prompt_set("human-eval")
    assert [item["id"] for item in humaneval] == [
        "HumanEval/23",
        "HumanEval/45",
        "HumanEval/53",
        "HumanEval/55",
    ]
    assert {item["id"]: len(item["text"]) for item in humaneval} == {
        "HumanEval/23": 133,
        "HumanEval/45": 138,
        "HumanEval/53": 115,
        "HumanEval/55": 130,
    }
    mt_bench = load_prompt_set("mtbench")
    assert [item["id"] for item in mt_bench] == ["81", "101", "111", "121"]
    assert mt_bench[0]["text"].startswith("Compose an engaging travel blog")
    assert "second person" in mt_bench[1]["text"]
    with pytest.raises(RuntimeError, match="CC BY-NC 4.0"):
        load_prompt_set("alpaca")


def test_tokenize_matches_the_tokenizer_and_caps_length() -> None:
    """Keep the start of the encoding when prompt_length is set."""
    tokenizer = _CharTokenizer()
    records = load_prompt_set("mt_bench")[:1]
    tensors = tokenize_prompts(records, tokenizer, torch.device("cpu"), 8)
    expected = tokenizer.encode(records[0]["text"])[:8]
    assert tensors[0].shape == (1, 8)
    assert tensors[0][0].tolist() == expected


def test_measurement_step_cap_uses_each_prompt_once() -> None:
    """A request larger than the fixed set does not repeat prompts."""
    _rows, ids, note = _prompts(
        {
            "dataset": "mt_bench",
            "model": "toy",
            "n_measurement_steps": 200,
        },
        torch.device("cpu"),
        _CharTokenizer(),
    )
    assert ids == ["81", "101", "111", "121"]
    assert len(_rows) == 4
    assert "n_measurement_steps asked for 200" in note
    assert "used once" in note
    assert "not a checkpoint measurement" in note


def test_synthetic_prompts_omit_ids() -> None:
    """An unknown dataset name stays on synthetic token ids."""
    rows, ids, note = _prompts(
        {
            "dataset": "unit",
            "n_samples": 1,
            "vocab_size": 8,
            "prompt_length": 2,
            "seed": 0,
        },
        torch.device("cpu"),
    )
    assert ids is None
    assert note == ""
    assert rows[0].shape == (1, 2)
    document = assemble_result(
        model="toy",
        method="flashspec_ucb",
        gamma=2,
        dataset="unit",
        n_samples=1,
        metrics={"tokens_per_second_mean": 1.0},
        measurement_kind="unit",
        notes="unit",
        unmeasured=[],
        hardware={"platform": "test"},
    )
    assert "prompt_ids" not in document


def test_toy_tokenizer_records_the_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record the fixed id and feed the tokenizer's ids to the engine."""
    seen: list[list[int]] = []
    original = SpeculativeEngine.generate

    def _spy(
        self: SpeculativeEngine,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> object:
        seen.append([int(item) for item in input_ids.detach().cpu().reshape(-1).tolist()])
        return original(self, input_ids, max_new_tokens=max_new_tokens)

    monkeypatch.setattr(SpeculativeEngine, "generate", _spy)
    tokenizer = _CharTokenizer()
    paired = run_paired_generations(_toy_config("human-eval"))
    spec = paired["flashspec_ucb"]
    assert spec["dataset"] == "humaneval"
    assert spec["prompt_ids"] == ["HumanEval/23"]
    assert paired["vanilla_ar"]["prompt_ids"] == ["HumanEval/23"]
    text = load_prompt_set("humaneval")[0]["text"]
    # The timed generation plus one untimed host-sync count, same prompt.
    assert seen == [tokenizer.encode(text)[:8]] * 2
    acceptance = spec["metrics"]["acceptance_rate_mean"]
    assert isinstance(acceptance, float)
    assert 0.0 <= acceptance <= 1.0
    assert "not a checkpoint measurement" in str(spec["notes"])


def test_llama_config_writes_nothing_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop the gated Llama config on the missing CUDA device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "out"
    with pytest.raises(DeviceUnavailableError, match="No result file"):
        run_config_file(Path("benchmarks/configs/llama3_8b.yaml"), output)
    assert list(output.glob("*.json")) == []


def test_alpaca_only_writes_nothing(tmp_path: Path) -> None:
    """Refuse a config whose only dataset is Alpaca."""
    config = tmp_path / "alpaca.yaml"
    config.write_text("model: toy\ndataset: alpaca\n", encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="CC BY-NC 4.0"):
        run_config_file(config, output)
    assert list(output.glob("*.json")) == []


def test_alpaca_skip_precedes_the_tokenizer_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skip Alpaca in order, then refuse the next real set with no tokenizer."""
    config = tmp_path / "mixed.yaml"
    config.write_text(
        "\n".join(
            [
                "model: toy",
                "device: cpu",
                "datasets:",
                "  - alpaca",
                "  - mt_bench",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="tokenizer"):
        run_config_file(config, output)
    assert list(output.glob("*.json")) == []
    assert "CC BY-NC 4.0" in capsys.readouterr().out


def test_result_names_keep_synthetic_files_distinct() -> None:
    """Include the dataset slug when a file would otherwise be overwritten."""
    synthetic = _result_filename("tiny_cpu", "synthetic_prompts", "flashspec_ucb", "abc")
    assert synthetic == "tiny_cpu_flashspec_ucb_abc.json"
    real = _result_filename("llama", "mt_bench", "vanilla_ar", "abc")
    assert real == "llama_mt_bench_vanilla_ar_abc.json"
    paired = _result_filename("listed", "unit", "flashspec_ucb", "abc", multiple=True)
    assert paired == "listed_unit_flashspec_ucb_abc.json"


def test_dataset_list_names_files_per_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Write one pair per runnable dataset and skip Alpaca."""
    seen: list[str] = []

    def _fake(config: dict[str, object]) -> dict[str, dict[str, object]]:
        seen.append(str(config["dataset"]))
        assert "datasets" not in config
        shared: dict[str, object] = {
            "model": "toy",
            "gamma": 2,
            "dataset": str(config["dataset"]),
            "n_samples": 1,
            "metrics": {"tokens_per_second_mean": 1.0},
            "measurement_kind": "unit",
            "notes": "unit",
            "unmeasured": [],
            "hardware": {"platform": "test"},
        }
        if config["dataset"] == "mt_bench":
            shared["prompt_ids"] = ["81"]
        return {
            "flashspec_ucb": assemble_result(method="flashspec_ucb", **shared),
            "vanilla_ar": assemble_result(method="vanilla_ar", **shared),
        }

    monkeypatch.setattr("benchmarks.harness.run_paired_generations", _fake)
    config = tmp_path / "listed.yaml"
    config.write_text(
        "\n".join(
            [
                "model: toy",
                "datasets:",
                "  - alpaca",
                "  - unit",
                "  - mt_bench",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    written = run_config_file(config, output)
    assert seen == ["unit", "mt_bench"]
    names = sorted(path.name for path in written)
    assert any(name.startswith("listed_unit_flashspec_ucb_") for name in names)
    assert any(name.startswith("listed_mt_bench_flashspec_ucb_") for name in names)
    assert all("alpaca" not in name for name in names)
    assert "CC BY-NC 4.0" in capsys.readouterr().out
