"""Phase 4 harness: chat prompts, neutral generation defaults, modes, per-method fields."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

import benchmarks.harness as harness
from benchmarks.harness import (
    _chat_ids,
    _hf_generate,
    _mode_slug,
    _neutralise_generation_config,
    _result_filename,
    run_config_file,
    run_paired_generations,
    summarise_runs,
)
from flashspec.metrics.instrument import NonFiniteLogits


def _llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=None,
        initializer_range=0.1,
    )
    return LlamaForCausalLM(config).eval()


class _ChatTokenizer:
    """Character tokenizer with a two-marker chat template."""

    chat_template = "fake"

    def encode(self, text: str) -> list[int]:
        """Map characters into a 64-id range."""
        return [3 + (ord(ch) % 60) for ch in text]

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> dict[str, list[int]]:
        """Return ``[1] + encode(content) + [2]`` as a BatchEncoding-like dict."""
        assert tokenize and add_generation_prompt
        return {"input_ids": [1, *self.encode(messages[0]["content"]), 2]}


def _toy_config(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401
    config: dict[str, Any] = {
        "model": "toy_llama",
        "device": "cpu",
        "gamma": 2,
        "max_new_tokens": 6,
        "n_warmup": 0,
        "n_samples": 3,
        "prompt_length": 6,
        "vocab_size": 128,
        "target_hidden": 32,
        "target_layers": 1,
        "drafter_hidden": 32,
        "drafter_layers": 1,
        "seed": 0,
        "dataset": "synthetic_prompts",
        "methods": ["flashspec_ucb", "hf_greedy", "hf_sample", "hf_assisted"],
    }
    config.update(overrides)
    return config


class TestGenerationDefaults:
    """Shipped sampling defaults must not leak into the plain baselines."""

    def test_shipped_defaults_change_greedy_until_neutralised(self) -> None:
        """A repetition penalty in the shipped config breaks greedy; neutralising restores it."""
        model = _llama()
        prompt = torch.randint(0, 128, (1, 6), generator=torch.Generator().manual_seed(1))
        reference = []
        with torch.no_grad():
            ids = prompt
            for _ in range(24):
                ids = torch.cat([ids, model(ids).logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        reference = ids[:, 6:]
        model.generation_config = GenerationConfig(
            do_sample=True, top_k=5, top_p=0.5, repetition_penalty=5.0, pad_token_id=0
        )
        shipped = _neutralise_generation_config(model)
        assert shipped["repetition_penalty"] == 5.0
        assert model.flashspec_generation_overrides == shipped  # type: ignore[comparison-overlap]
        assert model.generation_config.pad_token_id == 0
        assert torch.equal(_hf_generate(model, prompt, 24, 1.0, do_sample=False), reference)

    def test_sampling_baseline_applies_only_temperature(self) -> None:
        """After neutralising, the first sampled step's scores are logits / T, all finite."""
        model = _llama()
        model.generation_config = GenerationConfig(top_k=3, top_p=0.2, pad_token_id=0)
        _neutralise_generation_config(model)
        prompt = torch.randint(0, 128, (1, 6), generator=torch.Generator().manual_seed(2))
        seen: list[torch.Tensor] = []
        original = model.generate

        def _capture(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            out = original(*args, output_scores=True, return_dict_in_generate=True, **kwargs)
            seen.append(out.scores[0])
            return out.sequences

        model.generate = _capture  # type: ignore[method-assign]
        _hf_generate(model, prompt, 1, 0.7, do_sample=True)
        with torch.no_grad():
            raw = model(prompt).logits[:, -1] / 0.7
        out = SimpleNamespace(scores=seen)
        assert bool(torch.isfinite(out.scores[0]).all())
        assert torch.allclose(out.scores[0], raw, atol=1e-4)

    def test_eos_ids_are_kept_for_stop_at_eos(self) -> None:
        """The shipped EOS ids are remembered even though generate no longer sees them."""
        model = _llama()
        model.generation_config = GenerationConfig(eos_token_id=[5, 6], pad_token_id=0)
        _neutralise_generation_config(model)
        assert model.flashspec_eos_token_ids == [5, 6]  # type: ignore[comparison-overlap]
        assert model.generation_config.eos_token_id is None


class TestPrompts:
    """Real prompt sets go through the chat template when asked."""

    def test_chat_ids_accepts_dict_and_list(self) -> None:
        """BatchEncoding-like and plain-list returns both work."""
        tokenizer = _ChatTokenizer()
        assert _chat_ids(tokenizer, "ab")[0] == 1
        listing = SimpleNamespace(chat_template="x", apply_chat_template=lambda *_a, **_k: [[7, 8]])
        assert _chat_ids(listing, "ab") == [7, 8]
        with pytest.raises(RuntimeError, match="no chat template"):
            _chat_ids(SimpleNamespace(chat_template=None), "ab")

    def test_chat_template_prompts_in_results(self) -> None:
        """The run records the chat format, prompt lengths and the drafter name."""
        tokenizer = _ChatTokenizer()
        config = _toy_config(
            dataset="mt_bench",
            tokenizer=tokenizer,
            chat_template=True,
            prompt_length=None,
            n_samples=2,
            temperature=0.0,
        )
        paired = run_paired_generations(config)
        spec = paired["flashspec_ucb"]
        assert spec["prompt_format"] == "chat_template"
        assert spec["drafter"] == "toy_llama"
        texts = [
            item["text"]
            for item in __import__("benchmarks.prompt_sets", fromlist=["x"]).load_prompt_set(
                "mt_bench"
            )[:2]
        ]
        assert spec["prompt_tokens"] == [len(text) + 2 for text in texts]
        assert "chat template" in str(spec["notes"])


class TestModesAndMethods:
    """Matching baselines per mode, rotation, and per-method fields."""

    def test_greedy_mode_fields(self) -> None:
        """Greedy runs hf_greedy (not hf_sample) and checks tokens against it."""
        paired = run_paired_generations(_toy_config(temperature=0.0))
        assert set(paired) == {"flashspec_ucb", "hf_greedy", "hf_assisted"}
        spec = paired["flashspec_ucb"]
        match = spec["greedy_token_match"]
        assert isinstance(match, dict)
        assert match["reference"] == "hf_greedy"
        assert match["prompts_matched"] == match["n_prompts"] == 3
        assert match["first_divergence"] == [None, None, None]
        assistant = paired["hf_assisted"]
        assert isinstance(assistant["greedy_token_match"], dict)
        assert "greedy_token_match" not in paired["hf_greedy"]
        assert "speedup_vs_ar_mean" in spec["unmeasured"]
        metrics = spec["metrics"]
        assert isinstance(metrics, dict)
        assert {"speedup_vs_hf_greedy_median", "speedup_vs_hf_assisted_median"} <= set(metrics)

    def test_sample_mode_fields(self) -> None:
        """T > 0 runs hf_sample (not hf_greedy) and assisted generation samples too."""
        paired = run_paired_generations(_toy_config(temperature=0.7))
        assert set(paired) == {"flashspec_ucb", "hf_sample", "hf_assisted"}
        assert "greedy_token_match" not in paired["flashspec_ucb"]
        assert "sampled (temperature 0.7)" in str(paired["hf_assisted"]["notes"])

    def test_every_method_has_counts_probes_and_quantiles(self) -> None:
        """forward_counts, host_syncs, nonfinite_logits and tok/s median/IQR on every method."""
        paired = run_paired_generations(_toy_config(temperature=0.0))
        for name, doc in paired.items():
            counts = doc["forward_counts"]
            assert isinstance(counts, dict)
            assert counts["tokens_emitted"] == 3 * 6, name
            assert counts["target_forwards"] >= 3
            host = doc["host_syncs"]
            assert isinstance(host, dict)
            assert host["counter"] == "cpu_tensor_read_proxy"
            bad = doc["nonfinite_logits"]
            assert isinstance(bad, dict)
            assert bad["target_entries"] == 0
            assert bad["target_forwards_checked"] >= 1
            metrics = doc["metrics"]
            assert isinstance(metrics, dict)
            assert metrics["tokens_per_second_p25"] <= metrics["tokens_per_second_median"]
            assert metrics["tokens_per_second_median"] <= metrics["tokens_per_second_p75"]
            assert doc["method_order"] == "rotated by one position per prompt"
        spec_counts = paired["flashspec_ucb"]["forward_counts"]
        assert isinstance(spec_counts, dict)
        assert spec_counts["target_forwards_per_step"] == 1.0
        assert paired["flashspec_ucb"]["host_syncs"]["per_step"] <= 2.0  # type: ignore[index]

    def test_method_order_rotates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each prompt starts with a different method."""
        order: list[str] = []
        original = harness._hf_generate

        def _spy(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
            order.append("assisted" if kwargs.get("assistant_model") is not None else "plain")
            return original(*args, **kwargs)

        monkeypatch.setattr(harness, "_hf_generate", _spy)
        run_paired_generations(_toy_config(temperature=0.0, methods=["hf_greedy", "hf_assisted"]))
        # 3 timed prompts x 2 HF methods, then 2 probes. FlashSpec rotates in between.
        # Methods are [flashspec, hf_greedy, hf_assisted], shifted by one per prompt.
        assert order[:6] == ["plain", "assisted", "plain", "assisted", "assisted", "plain"]

    def test_stop_at_eos_needs_eos_ids(self) -> None:
        """stop_at_eos on a target with no EOS id refuses to run."""
        with pytest.raises(RuntimeError, match="eos_token_id"):
            run_paired_generations(_toy_config(temperature=0.0, stop_at_eos=True))


class TestConfigs:
    """Temperatures loop, file names, and the checked-in T4 configs."""

    def test_mode_slug_and_filename(self) -> None:
        """Mode goes after the dataset slug."""
        assert _mode_slug(0.0) == "greedy"
        assert _mode_slug(0.7) == "t0.7"
        name = _result_filename("qwen_t4", "mt_bench", "hf_sample", "abc1234", mode="t0.7")
        assert name == "qwen_t4_mt_bench_t0.7_hf_sample_abc1234.json"

    def test_temperatures_write_one_file_per_mode_and_method(self, tmp_path: Path) -> None:
        """A toy config with two temperatures writes greedy and sampled files."""
        config_path = tmp_path / "modes.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model: toy_llama",
                    "device: cpu",
                    "gamma: 2",
                    "max_new_tokens: 4",
                    "n_warmup: 0",
                    "n_samples: 1",
                    "prompt_length: 5",
                    "vocab_size: 128",
                    "target_hidden: 32",
                    "target_layers: 1",
                    "drafter_hidden: 32",
                    "drafter_layers: 1",
                    "dataset: synthetic_prompts",
                    "temperatures:",
                    "  - 0.0",
                    "  - 0.7",
                    "methods:",
                    "  - flashspec_ucb",
                    "  - hf_greedy",
                    "  - hf_sample",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        written = run_config_file(config_path, tmp_path / "out")
        names = sorted(path.name.rsplit("_", 1)[0] for path in written)
        assert names == [
            "modes_synthetic_prompts_greedy_flashspec_ucb",
            "modes_synthetic_prompts_greedy_hf_greedy",
            "modes_synthetic_prompts_t0.7_flashspec_ucb",
            "modes_synthetic_prompts_t0.7_hf_sample",
        ]

    @pytest.mark.parametrize(
        ("name", "target", "drafter"),
        [
            ("qwen_t4", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
            ("tinyllama_t4", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Felladrin/Llama-160M-Chat-v1"),
        ],
    )
    def test_t4_configs(self, name: str, target: str, drafter: str) -> None:
        """The Phase 4 configs ask for exactly the agreed run."""
        config = harness.load_benchmark_config(Path(f"benchmarks/configs/{name}.yaml"))
        assert config["model"] == target
        assert config["drafter"] == drafter
        assert config["dtype"] == "float16"
        assert config["batch_size"] == 1
        assert config["gamma"] == 4
        assert config["max_new_tokens"] == 128
        assert config["n_warmup"] == 2
        assert config["n_samples"] == 10
        assert config["chat_template"] is True
        assert config["datasets"] == ["mt_bench", "humaneval"]
        assert config["temperatures"] == [0.0, 0.7]
        assert config["methods"] == ["flashspec_ucb", "hf_greedy", "hf_sample", "hf_assisted"]
        assert "synthetic" not in str(config.get("dataset", ""))


def test_shared_tokenizer_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal vocab sizes are not enough; differing ids refuse the pair."""
    vocabularies = {"a": {"x": 0, "y": 1}, "b": {"x": 1, "y": 0}, "c": {"x": 0, "y": 1}}
    monkeypatch.setattr(
        harness,
        "_load_tokenizer",
        lambda name: SimpleNamespace(get_vocab=lambda: vocabularies[name]),
    )
    harness._assert_shared_tokenizer("a", "c")
    with pytest.raises(RuntimeError, match="differ on 2 entries"):
        harness._assert_shared_tokenizer("a", "b")


def test_nonfinite_logits_counter() -> None:
    """NaN and inf entries are counted; finite logits count zero."""
    model = _llama()
    with NonFiniteLogits(model) as counter, torch.no_grad():
        model(torch.zeros(1, 3, dtype=torch.long))
    assert counter.count == 0
    assert counter.calls == 1
    poisoned = _llama()
    with torch.no_grad():
        poisoned.lm_head.weight[0, 0] = float("nan")
    with NonFiniteLogits(poisoned) as counter, torch.no_grad():
        poisoned(torch.zeros(1, 2, dtype=torch.long))
    assert counter.count > 0


def test_summarise_runs_quantiles() -> None:
    """Median and IQR use linear interpolation between order statistics."""
    metrics = summarise_runs([1000.0, 500.0, 250.0, 125.0], [10, 10, 10, 10], None)
    assert metrics["tokens_per_second_median"] == 30.0
    assert metrics["tokens_per_second_p25"] == 17.5
    assert metrics["tokens_per_second_p75"] == 50.0
    assert metrics["tokens_per_second_iqr"] == 32.5
    single = summarise_runs([100.0], [5], None)
    assert single["tokens_per_second_p25"] == single["tokens_per_second_p75"] == 50.0


class TestPhase5Methods:
    """flashspec_compiled, compiled HF baselines and the forward-cost probe (CPU: not compiled)."""

    def test_compiled_methods_and_cost_fields(self) -> None:
        """Every method writes a document; compiled variants say whether compile was effective."""
        methods = [
            "flashspec_ucb",
            "flashspec_compiled",
            "hf_greedy",
            "hf_greedy_compiled",
            "hf_assisted",
        ]
        paired = run_paired_generations(_toy_config(temperature=0.0, methods=methods))
        assert set(paired) == set(methods)
        for name in ("flashspec_ucb", "flashspec_compiled"):
            doc = paired[name]
            cost = doc["step_cost"]
            assert isinstance(cost, dict)
            assert cost["c"] is not None and cost["c"] > 0
            match = doc["greedy_token_match"]
            assert isinstance(match, dict)
            assert match["prompts_matched"] == match["n_prompts"] == 3
            metrics = doc["metrics"]
            assert isinstance(metrics, dict)
            assert "speedup_vs_hf_greedy_compiled_median" in metrics
        compiled = paired["flashspec_compiled"]
        expected = {
            "requested": True,
            "effective": False,
            "decision": "compile=True needs a CUDA device for CUDA graphs; the device is cpu",
            "fallback_reason": None,
        }
        assert compiled["compile"] == expected
        assert "speedup_vs_flashspec_ucb_median" in compiled["metrics"]  # type: ignore[operator]
        hf_compiled = paired["hf_greedy_compiled"]
        assert hf_compiled["compile"] == {"requested": True, "effective": False}
        assert hf_compiled["greedy_token_match"]["prompts_matched"] == 3  # type: ignore[index]
        counts = hf_compiled["forward_counts"]
        assert isinstance(counts, dict)
        assert str(counts["source"]).startswith("derived")
        assert "not checked" in str(hf_compiled["nonfinite_logits"])
        for doc in paired.values():
            forward_cost = doc["forward_cost"]
            assert isinstance(forward_cost, dict)
            eager = forward_cost["eager"]
            assert isinstance(eager, dict)
            assert eager["c"] is not None and eager["c"] > 0
            assert forward_cost["compiled"] is None

    def test_flashspec_compile_auto_is_recorded(self) -> None:
        """flashspec_compile: auto runs the default-user decision and records its reason."""
        methods = ["flashspec_ucb", "flashspec_compiled", "hf_greedy"]
        config = _toy_config(temperature=0.0, methods=methods, flashspec_compile="auto")
        compiled = run_paired_generations(config)["flashspec_compiled"]["compile"]
        assert isinstance(compiled, dict)
        assert compiled["requested"] == "auto"
        assert compiled["effective"] is False
        assert "needs a CUDA device" in str(compiled["decision"])

    def test_compile_request_values(self) -> None:
        """The config key defaults to True and accepts only True or 'auto'."""
        assert harness._compile_request({}) is True
        assert harness._compile_request({"flashspec_compile": True}) is True
        assert harness._compile_request({"flashspec_compile": "auto"}) == "auto"
        for bad in (False, "on", 1):
            with pytest.raises(ValueError, match="flashspec_compile"):
                harness._compile_request({"flashspec_compile": bad})

    def test_sampled_mode_runs_hf_sample_compiled(self) -> None:
        """At T > 0 the compiled plain baseline is hf_sample_compiled."""
        methods = ["flashspec_ucb", "hf_sample", "hf_sample_compiled", "hf_greedy_compiled"]
        paired = run_paired_generations(_toy_config(temperature=0.7, methods=methods))
        assert set(paired) == {"flashspec_ucb", "hf_sample", "hf_sample_compiled"}
