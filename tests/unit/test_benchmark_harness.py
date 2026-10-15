"""Measurement harness: no fake zeros, hardware labelled, toy e2e actually runs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks.harness import (
    DeviceUnavailableError,
    _assert_shared_vocab,
    _TinyCausalLM,
    _torch_dtype_from_config,
    assemble_result,
    load_benchmark_config,
    percentile,
    resolve_device,
    run_paired_generations,
    summarise_runs,
)
from scripts.update_benchmark_results import _validate_result_file


def test_percentile_and_summary_omit_unmeasured_acceptance() -> None:
    """Vanilla AR summaries do not invent an acceptance rate."""
    assert percentile([10.0, 30.0, 20.0], 50) == 20.0
    metrics = summarise_runs([10.0, 30.0], [4, 4], None)
    assert "acceptance_rate_mean" not in metrics
    assert metrics["tokens_per_second_mean"] > 0
    assert "p95_latency_ms" in metrics


def test_cuda_device_without_cuda_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA config on a CPU machine raises and does not imply a result."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="No result file"):
        resolve_device("cuda:0")
    assert resolve_device("auto").type == "cpu"


def test_simple_yaml_and_template_file() -> None:
    """The checked-in tiny config parses, and baseline.json is not a measurement."""
    config = load_benchmark_config(Path("benchmarks/configs/tiny_cpu.yaml"))
    assert config["model"] == "toy"
    assert config["methods"] == ["flashspec_ucb", "vanilla_ar"]
    errors = _validate_result_file(Path("benchmarks/results/baseline.json"))
    assert errors == []


def test_toy_e2e_writes_only_measured_fields(tmp_path: Path) -> None:
    """A tiny in-process run records throughput, latency, and acceptance."""
    config = {
        "model": "toy",
        "device": "cpu",
        "gamma": 2,
        "max_new_tokens": 2,
        "n_warmup": 0,
        "n_samples": 1,
        "prompt_length": 4,
        "vocab_size": 32,
        "target_hidden": 32,
        "target_layers": 1,
        "drafter_hidden": 32,
        "drafter_layers": 1,
        "seed": 0,
        "dataset": "unit",
    }
    paired = run_paired_generations(config)
    spec = paired["flashspec_ucb"]
    ar = paired["vanilla_ar"]
    assert spec["status"] == "measured"
    assert spec["measurement_kind"] == "toy_random_init_transformer"
    assert "gpu" not in spec["hardware"] or torch.cuda.is_available()
    spec_metrics = spec["metrics"]
    assert isinstance(spec_metrics, dict)
    assert spec_metrics["tokens_per_second_mean"] > 0
    assert "acceptance_rate_mean" in spec_metrics
    assert "speedup_vs_ar_mean" in spec_metrics
    ar_metrics = ar["metrics"]
    assert isinstance(ar_metrics, dict)
    assert "acceptance_rate_mean" not in ar_metrics
    assert "acceptance_rate_mean" in ar["unmeasured"]
    # Speedup is whatever this machine measured, including values below 1.
    assert spec_metrics["speedup_vs_ar_mean"] > 0


def test_all_zero_placeholder_is_rejected(tmp_path: Path) -> None:
    """A zero-filled file that claims to be a result fails validation."""
    path = tmp_path / "fake.json"
    path.write_text(
        """{
          "timestamp": "1970-01-01T00:00:00Z",
          "git_commit": "pending",
          "hardware": {"gpu": "NVIDIA H100 SXM5 80GB"},
          "model": "meta-llama/Llama-3-8B-Instruct",
          "method": "flashspec_ucb",
          "gamma": 4,
          "dataset": "mt_bench",
          "n_samples": 0,
          "metrics": {
            "tokens_per_second_mean": 0.0,
            "tokens_per_second_std": 0.0,
            "speedup_vs_ar_mean": 0.0,
            "speedup_vs_ar_std": 0.0,
            "acceptance_rate_mean": 0.0,
            "acceptance_rate_std": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "gpu_memory_gb": 0.0
          }
        }""",
        encoding="utf-8",
    )
    errors = _validate_result_file(path)
    assert errors


def test_dtype_names_and_vocab_guard() -> None:
    """Checkpoint dtype names are explicit, and a vocab mismatch writes nothing."""
    assert _torch_dtype_from_config({}) is torch.float32
    assert _torch_dtype_from_config({"dtype": "float16"}) is torch.float16
    assert _torch_dtype_from_config({"dtype": "bf16"}) is torch.bfloat16
    with pytest.raises(RuntimeError, match="No result file"):
        _torch_dtype_from_config({"dtype": "int8"})
    _assert_shared_vocab(_TinyCausalLM(32, 32, 1), _TinyCausalLM(32, 32, 1))
    with pytest.raises(RuntimeError, match="No result file"):
        _assert_shared_vocab(_TinyCausalLM(32, 32, 1), _TinyCausalLM(64, 32, 1))


def test_assemble_result_round_trip_keys() -> None:
    """Measured documents carry status and do not require a GPU name."""
    document = assemble_result(
        model="toy",
        method="flashspec_ucb",
        gamma=2,
        dataset="unit",
        n_samples=1,
        metrics={"tokens_per_second_mean": 1.0},
        measurement_kind="unit",
        notes="unit",
        unmeasured=["gpu_memory_gb"],
        hardware={"platform": "test", "avx2": False},
    )
    assert document["status"] == "measured"
    assert document["hardware"] == {"platform": "test", "avx2": False}
