"""Phase 6 measurement: every fixed arm, the bandits and HuggingFace, per set and mixed.

A config with ``runner: phase6`` is dispatched here by
:func:`benchmarks.harness.run_config_file` (so ``benchmarks/gpu_bundle.py``
and the notebook use it unchanged). See
``benchmarks/configs/qwen_t4_phase6.yaml`` for the keys.

What it runs, per block (a prompt set) and per temperature:

- ``fixed_arms``: one engine per arm in ``arms`` (``arm_<label>``), alone.
- ``bandits``: one engine over all arms per selector (``bandit_ucb1``,
  ``bandit_thompson``). A block runs its prompts in order on one engine,
  so the bandit's state carries across prompts; a new block starts fresh.
- ``flashspec_compiled_legacy`` / ``bandit_ucb1_legacy``: the γ = 4 model
  arm, and the UCB1 bandit, with ``compile_fixed_shapes=False`` (the 0.2.0
  call shapes). They are the "before" of the CUDA-graph count.
- HuggingFace baselines: ``hf_greedy`` (eager), ``hf_greedy_compiled``,
  ``hf_sample_compiled``, ``hf_assisted``.

Engines whose arms have the same ``gamma_max`` share one target wrapper
(one static cache, one set of CUDA graphs); all fixed-shape engines share
one drafter wrapper (their drafter calls use the same widths); the legacy
methods have their own of each. Every FlashSpec method therefore verifies
at the width a user running that arm set alone would get.

Each method's document has the usual harness fields plus ``arms``,
``arm_report`` (:func:`benchmarks.arm_report.arm_report`, from the measured
generations), ``cuda_graphs`` (captures and call shapes of its group),
``outputs`` with sha256 per prompt, ``greedy_token_match`` (vs eager
``hf_greedy``) and ``greedy_token_match_compiled`` (vs ``hf_greedy_compiled``)
in greedy mode, and ``metrics.workload_tokens_per_second`` (total tokens /
total seconds over the block). The mixed block takes its greedy references
from the per-set blocks by prompt id when it does not run HuggingFace
itself, so list the per-set blocks first.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from benchmarks.arm_report import arm_report
from benchmarks.harness import (
    _CAPTURES,
    _COLD_STARTED,
    TOY_MODELS,
    _assert_shared_vocab,
    _build_toy_llama_modules,
    _hf_generate,
    _load_checkpoint_modules,
    _load_tokenizer,
    _mode_slug,
    _module_vocab,
    _prompts,
    _uses_hf_generate,
    assemble_result,
    detect_hardware,
    git_commit,
    graph_report,
    load_benchmark_config,
    output_records,
    resolve_device,
    summarise_runs,
    write_result,
)
from benchmarks.prompt_sets import REAL_DATASETS, canonical_dataset, load_prompt_set
from flashspec.bandit.base import DraftSelector
from flashspec.bandit.thompson import ThompsonSelector
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import ArmSpec, ArmStep
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import count_host_syncs
from flashspec.metrics.step_profile import StepRecord, summarise_steps
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig
from flashspec.utils.device import set_seed

__all__ = ["run_phase6_config"]

_MODEL = 0  # drafter index of the model drafter in every engine
_NGRAM = 1  # drafter index of the lookup drafter
_DRAFTERS = {"model": _MODEL, "ngram": _NGRAM}
_HF_METHODS = ("hf_greedy", "hf_greedy_compiled", "hf_sample_compiled", "hf_assisted")
_LEGACY_GAMMA = 4


class _CharTokenizer:
    """Toy runs only: one token per character, ids 1..50."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [(ord(character) % 50) + 1 for character in text]


@dataclass
class _Group:
    """Wrappers shared by engines with the same verify width (or one legacy method)."""

    name: str
    target: TargetModel
    drafter: HuggingFaceDrafter
    ngram: NgramDrafter
    methods: list[str] = field(default_factory=list)

    def drafters(self) -> list[Any]:
        return [self.drafter, self.ngram]


@dataclass
class _Method:
    """One FlashSpec method: its arms, selector kind and wrapper group."""

    name: str
    arms: list[ArmSpec]
    selector: str  # "fixed", "ucb1" or "thompson"
    fixed_shapes: bool
    group: _Group | None = None


@dataclass
class _Runs:
    latency_ms: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    outputs: list[list[int]] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    arm_steps: list[list[ArmStep]] = field(default_factory=list)


def _parse_arms(config: dict[str, Any]) -> list[ArmSpec]:
    raw = config.get("arms")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("A phase6 config needs a non-empty 'arms' list.")
    arms: list[ArmSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError(f"Each arm must be a mapping; got {item!r}.")
        drafter = _DRAFTERS[str(item["drafter"])]
        fallback = item.get("fallback")
        arms.append(
            ArmSpec(
                drafter=drafter,
                gamma=int(item["gamma"]),
                label=str(item["label"]),
                fallback=_DRAFTERS[str(fallback)] if fallback is not None else None,
            )
        )
    labels = [arm.label for arm in arms]
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"Arm labels must be unique; got {labels}.")
    return arms


def _expand(names: list[str], arms: list[ArmSpec], bandits: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name == "fixed_arms":
            out += [f"arm_{arm.label}" for arm in arms]
        elif name == "bandits":
            out += [f"bandit_{kind}" for kind in bandits]
        else:
            out.append(name)
    return out


def _flashspec_methods(names: list[str], arms: list[ArmSpec]) -> dict[str, _Method]:
    by_label = {f"arm_{arm.label}": arm for arm in arms}
    methods: dict[str, _Method] = {}
    for name in names:
        if name in by_label:
            methods[name] = _Method(name, [by_label[name]], "fixed", True)
        elif name in ("bandit_ucb1", "bandit_thompson"):
            methods[name] = _Method(name, list(arms), name.removeprefix("bandit_"), True)
        elif name == "bandit_ucb1_legacy":
            methods[name] = _Method(name, list(arms), "ucb1", False)
        elif name == "flashspec_compiled_legacy":
            legacy_arm = ArmSpec(_MODEL, _LEGACY_GAMMA, label=f"model-g{_LEGACY_GAMMA}")
            methods[name] = _Method(name, [legacy_arm], "fixed", False)
        elif name not in _HF_METHODS:
            raise RuntimeError(f"Unknown phase6 method {name!r}.")
    return methods


def _selector(kind: str, n_arms: int, window: int, seed: int) -> DraftSelector:
    if kind == "thompson":
        return ThompsonSelector(n_arms=n_arms, window_size=window, seed=seed)
    return UCB1Selector(n_arms=n_arms, window_size=window)


def run_phase6_config(
    path: Path,
    output_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    modules: tuple[nn.Module, nn.Module] | None = None,
) -> list[Path]:
    """Run a ``runner: phase6`` config and write one JSON per block, mode and method.

    Parameters
    ----------
    path : Path
        Config YAML; its stem names the result files.
    output_dir : Path
        Directory for the JSON results.
    config : dict or None
        Already-loaded config (tests); default loads ``path``.
    modules : tuple or None
        ``(drafter, target)`` modules to use instead of loading (tests).

    Returns
    -------
    list[Path]
        Files written.
    """
    config = dict(config if config is not None else load_benchmark_config(path))
    device = resolve_device(str(config.get("device", "auto")))
    model_name = str(config.get("model"))
    toy = model_name in TOY_MODELS
    seed = int(config.get("seed", 0))
    set_seed(seed)
    max_new_tokens = int(config.get("max_new_tokens", 128))
    n_warmup = int(config.get("n_warmup", 2))
    window = int(config.get("bandit_window", BanditConfig().window_size))
    compile_request = config.get("compile", "auto")
    if compile_request not in (True, False, "auto"):
        raise RuntimeError(f"compile must be true, false or 'auto'; got {compile_request!r}.")
    temperatures = [float(item) for item in config.get("temperatures", [0.0])]
    arms = _parse_arms(config)
    bandits = [str(item) for item in config.get("bandits", ["ucb1", "thompson"])]
    blocks = config.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise RuntimeError("A phase6 config needs a non-empty 'blocks' list.")

    if modules is not None:
        drafter_module, target_module = modules
    elif model_name == "toy_llama":
        drafter_module, target_module = _build_toy_llama_modules(config, device)
    else:
        drafter_module, target_module = _load_checkpoint_modules(config, device)
    _assert_shared_vocab(drafter_module, target_module)
    tokenizer = config.get("tokenizer")
    if tokenizer is None:
        tokenizer = _CharTokenizer() if toy else _load_tokenizer(model_name)
    weight_dtype = next(target_module.parameters()).dtype
    vocab = _module_vocab(target_module, config)

    # Prompts per block, and the cache length every static session needs.
    block_prompts: list[tuple[str, list[str], list[torch.Tensor], str]] = []
    for block in blocks:
        dataset = canonical_dataset(str(block["dataset"]))
        if dataset not in REAL_DATASETS:
            raise RuntimeError(f"phase6 blocks use fixed prompt sets; got {dataset!r}.")
        # Every prompt of the set, in its stored order.
        one = {
            **config,
            "dataset": dataset,
            "vocab_size": vocab,
            "n_samples": len(load_prompt_set(dataset)),
        }
        prompts, ids, note = _prompts(one, device, tokenizer)
        assert ids is not None
        if "n_prompts" in block:
            # Smoke tests only; the measurement configs run every prompt.
            limit = int(block["n_prompts"])
            prompts, ids = prompts[:limit], ids[:limit]
            note += f" Only the first {limit} prompts of the set (n_prompts)."
        block_prompts.append((dataset, ids, prompts, note))
    longest = max(
        int(prompt.shape[1]) for _d, _i, prompts, _n in block_prompts for prompt in prompts
    )
    gamma_max = max(arm.gamma for arm in arms)
    cache_len = max(
        int(config.get("compile_max_cache_len", 1024)),
        longest + max_new_tokens + 2 * (gamma_max + 1) + 64,
    )

    all_names = [_expand(list(block["methods"]), arms, bandits) for block in blocks]
    spec_methods: dict[str, _Method] = {}
    for names in all_names:
        spec_methods.update(_flashspec_methods(names, arms))
    groups: dict[str, _Group] = {}
    # Fixed-shape groups differ only in the target's verify width; their
    # drafter calls use the same widths, so they share one drafter wrapper.
    shared_drafter: HuggingFaceDrafter | None = None
    for method in spec_methods.values():
        width = max(arm.gamma for arm in method.arms) + 1
        key = f"verify_width_{width}" if method.fixed_shapes else f"legacy_{method.name}"
        if key not in groups:
            drafter = HuggingFaceDrafter(
                str(config.get("drafter", model_name)), device=device, model=drafter_module
            )
            if method.fixed_shapes:
                shared_drafter = shared_drafter or drafter
                drafter = shared_drafter
            groups[key] = _Group(
                name=key,
                target=TargetModel(model_name, device, weight_dtype, model=target_module),
                drafter=drafter,
                ngram=NgramDrafter(),
            )
        method.group = groups[key]
        groups[key].methods.append(method.name)

    hf_cache: Any = None
    if any(name.endswith("_compiled") and name.startswith("hf_") for n in all_names for name in n):
        from transformers import StaticCache

        hf_cache = StaticCache(config=target_module.config, max_cache_len=cache_len)

    hardware = detect_hardware()
    commit = git_commit()
    written: list[Path] = []
    # Greedy references by (temperature, dataset, prompt id), for the mixed block.
    references: dict[tuple[float, str, str], dict[str, list[int]]] = {}
    for (dataset, ids, prompts, note), names in zip(block_prompts, all_names, strict=True):
        for temperature in temperatures:
            print(f"phase6: dataset={dataset} mode={_mode_slug(temperature)}")
            documents = _run_block(
                config=config,
                dataset=dataset,
                prompt_ids=ids,
                prompts=prompts,
                prompt_note=note,
                names=names,
                spec_methods=spec_methods,
                groups=groups,
                temperature=temperature,
                device=device,
                model_name=model_name,
                drafter_module=drafter_module,
                target_module=target_module,
                hf_cache=hf_cache,
                cache_len=cache_len,
                compile_request=compile_request,
                max_new_tokens=max_new_tokens,
                n_warmup=n_warmup,
                window=window,
                seed=seed,
                toy=toy,
                hardware=hardware,
                references=references,
            )
            for method, document in documents.items():
                destination = output_dir / (
                    f"{path.stem}_{dataset}_{_mode_slug(temperature)}_{method}_{commit}.json"
                )
                write_result(destination, document)
                written.append(destination)
                metrics = document["metrics"]
                assert isinstance(metrics, dict)
                print(
                    f"{method}: {metrics.get('workload_tokens_per_second')} tok/s "
                    "(block total tokens / total seconds)"
                )
    return written


def _engine(
    method: _Method,
    *,
    temperature: float,
    seed: int,
    window: int,
    compile_request: Any,  # noqa: ANN401
    cache_len: int,
    max_new_tokens: int,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> SpeculativeEngine:
    assert method.group is not None
    n_arms = len(method.arms)
    config = FlashSpecConfig(
        device=str(device),
        dtype=str(weight_dtype).removeprefix("torch."),  # type: ignore[arg-type]
        max_new_tokens=max_new_tokens,
        compile=compile_request,
        compile_max_cache_len=cache_len,
        compile_fixed_shapes=method.fixed_shapes,
        bandit=BanditConfig(
            n_arms=n_arms,
            strategy="thompson" if method.selector == "thompson" else "ucb1",
            window_size=window,
        ),
        sampling=SamplingConfig(gamma=method.arms[0].gamma, temperature=temperature, seed=seed),
        metrics=MetricsConfig(profile_steps=True),
    )
    return SpeculativeEngine(
        config,
        method.group.drafters(),
        method.group.target,
        _selector(method.selector, n_arms, window, seed),
        arms=method.arms,
    )


def _run_block(
    *,
    config: dict[str, Any],
    dataset: str,
    prompt_ids: list[str],
    prompts: list[torch.Tensor],
    prompt_note: str,
    names: list[str],
    spec_methods: dict[str, _Method],
    groups: dict[str, _Group],
    temperature: float,
    device: torch.device,
    model_name: str,
    drafter_module: nn.Module,
    target_module: nn.Module,
    hf_cache: Any,  # noqa: ANN401
    cache_len: int,
    compile_request: Any,  # noqa: ANN401
    max_new_tokens: int,
    n_warmup: int,
    window: int,
    seed: int,
    toy: bool,
    hardware: dict[str, object],
    references: dict[tuple[float, str, str], dict[str, list[int]]],
) -> dict[str, dict[str, object]]:
    greedy = temperature == 0.0
    weight_dtype = next(target_module.parameters()).dtype
    for group in groups.values():
        group.drafter.set_temperature(temperature)

    runnable: list[str] = []
    for name in names:
        if name in spec_methods:
            runnable.append(name)
            continue
        if not _uses_hf_generate(target_module):
            print(f"Skipping {name}: needs HuggingFace modules with generate().")
            continue
        if name in ("hf_greedy", "hf_greedy_compiled") and not greedy:
            continue
        if name == "hf_sample_compiled" and greedy:
            continue
        runnable.append(name)

    def make(name: str) -> SpeculativeEngine:
        return _engine(
            spec_methods[name],
            temperature=temperature,
            seed=seed,
            window=window,
            compile_request=compile_request,
            cache_len=cache_len,
            max_new_tokens=max_new_tokens,
            device=device,
            weight_dtype=weight_dtype,
        )

    block_id = f"{dataset}:{temperature}"

    def generate(name: str, engine: SpeculativeEngine | None, prompt: torch.Tensor) -> Any:  # noqa: ANN401
        if engine is not None:
            return engine.generate(prompt, max_new_tokens=max_new_tokens)
        return _hf_generate(
            target_module,
            prompt,
            max_new_tokens,
            temperature,
            do_sample=not greedy,
            assistant_model=drafter_module if name == "hf_assisted" else None,
            static_cache=hf_cache if name.endswith("_compiled") else None,
        )

    def timed(
        name: str, engine: SpeculativeEngine | None, prompt: torch.Tensor
    ) -> tuple[Any, float]:  # noqa: ANN401
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with _CAPTURES.section(f"{block_id}:{name}"):
            out = generate(name, engine, prompt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return out, (time.perf_counter() - started) * 1000.0

    cold_start: dict[str, dict[str, object]] = {}
    # Warmup: compile and capture in throwaway engines, so the measured
    # bandits start with no state.
    for prompt in prompts[: max(n_warmup, 0)]:
        for name in runnable:
            engine = make(name) if name in spec_methods else None
            out, elapsed = timed(name, engine, prompt)
            _note_cold(cold_start, name, elapsed, out, model_name, config, warmup=True)

    engines = {name: make(name) for name in runnable if name in spec_methods}
    runs = {name: _Runs() for name in runnable}
    for index, prompt in enumerate(prompts):
        shift = index % len(runnable)
        for name in runnable[shift:] + runnable[:shift]:
            out, elapsed = timed(name, engines.get(name), prompt)
            _note_cold(cold_start, name, elapsed, out, model_name, config, warmup=False)
            stats = runs[name]
            stats.latency_ms.append(elapsed)
            if name in engines:
                produced = out.output_ids[out.output_ids >= 0].view(-1).tolist()
                stats.alpha.append(float(out.acceptance_rate))
                stats.steps.extend(out.step_profile or ())
                stats.arm_steps.append(list(out.arm_steps))
            else:
                produced = out.view(-1).tolist()
            stats.tokens.append(len(produced))
            stats.outputs.append([int(token) for token in produced])

    # Untimed probes: host reads per step, on a throwaway engine.
    counter_kind = "cuda_sync_debug_mode" if device.type == "cuda" else "cpu_tensor_read_proxy"
    probes: dict[str, dict[str, object]] = {}
    for name in engines:
        probe_engine = make(name)
        with count_host_syncs(device) as syncs:
            result = probe_engine.generate(prompts[0], max_new_tokens=max_new_tokens)
        n_steps = len(result.arm_steps)
        probes[name] = {
            "counter": counter_kind,
            "prompt_id": prompt_ids[0],
            "total": syncs.count,
            "n_steps": n_steps,
            "per_step": syncs.count / n_steps if n_steps else 0.0,
            "by_kind": dict(syncs.by_kind),
        }

    # Greedy references for this block, then for the mixed block by id.
    if greedy:
        for ref in ("hf_greedy", "hf_greedy_compiled"):
            if ref in runs:
                for pid, tokens in zip(prompt_ids, runs[ref].outputs, strict=True):
                    key = (temperature, dataset, pid)
                    references.setdefault(key, {})[ref] = tokens
                    if dataset != "mixed":
                        references.setdefault((temperature, "mixed", f"{dataset}/{pid}"), {})[
                            ref
                        ] = tokens

    def match(stats: _Runs, ref: str) -> dict[str, object] | None:
        wanted = [references.get((temperature, dataset, pid), {}).get(ref) for pid in prompt_ids]
        if any(item is None for item in wanted):
            return None
        firsts: list[int | None] = []
        for got, want in zip(stats.outputs, wanted, strict=True):
            assert want is not None
            if got == want:
                firsts.append(None)
                continue
            limit = min(len(got), len(want))
            firsts.append(next((i for i in range(limit) if got[i] != want[i]), limit))
        return {
            "reference": ref,
            "prompts_matched": sum(item is None for item in firsts),
            "n_prompts": len(firsts),
            "first_divergence": firsts,
            "source": "this block" if ref in runs else "per-set blocks, by prompt id",
        }

    def paired(a: _Runs, b: _Runs) -> float:
        ratios = [
            (an / (am / 1000.0)) / (bn / (bm / 1000.0))
            for am, an, bm, bn in zip(a.latency_ms, a.tokens, b.latency_ms, b.tokens, strict=True)
            if am > 0.0 and bm > 0.0 and bn > 0
        ]
        return statistics.median(ratios) if ratios else 0.0

    def workload(stats: _Runs) -> float:
        seconds = sum(stats.latency_ms) / 1000.0
        return sum(stats.tokens) / seconds if seconds > 0.0 else 0.0

    common_notes = (
        f"Phase 6: {dataset} prompts, {'greedy' if greedy else f'temperature {temperature}'}, "
        f"{max_new_tokens} new tokens each, no EOS stop. Methods run in an order rotated by one "
        "position per prompt; bandit engines run the block's prompts in order on one engine."
        + prompt_note
    )
    if toy:
        common_notes += (
            " Randomly initialised toy models: a CPU correctness run, not a speed claim."
        )
    documents: dict[str, dict[str, object]] = {}
    for name in runnable:
        stats = runs[name]
        is_spec = name in engines
        metrics = summarise_runs(stats.latency_ms, stats.tokens, stats.alpha if is_spec else None)
        metrics["workload_tokens_per_second"] = workload(stats)
        for other in runnable:
            if other != name:
                metrics[f"speedup_vs_{other}_median"] = paired(stats, runs[other])
                metrics[f"workload_speedup_vs_{other}"] = (
                    workload(stats) / workload(runs[other]) if workload(runs[other]) else 0.0
                )
        extra: dict[str, object] = {
            "phase6": {"block": dataset, "temperature": temperature, "methods": runnable},
            "cold_start": cold_start.get(name),
            "decoding": {
                "mode": "greedy" if greedy else "sample",
                "temperature": temperature,
                "stop_at_eos": False,
                "max_new_tokens": max_new_tokens,
            },
            "drafter": str(config.get("drafter", model_name)),
            "prompt_format": "chat_template" if config.get("chat_template") else "raw",
            "prompt_tokens": [int(prompt.shape[1]) for prompt in prompts],
            "method_order": "rotated by one position per prompt",
            "outputs": output_records(stats.outputs, prompt_ids),
            "forward_counts": _forward_counts(stats, is_spec),
            "nonfinite_logits": {
                "scope": "not checked: a forward hook would force the compiled forward to "
                "recompile",
                "prompt_id": prompt_ids[0],
            },
            "host_syncs": {
                "scope": "not probed for HuggingFace baselines in the phase6 runner",
            },
        }
        if is_spec:
            method = spec_methods[name]
            engine = engines[name]
            assert method.group is not None
            extra["arms"] = [
                {
                    "label": arm.label,
                    "drafter": "ngram" if arm.drafter == _NGRAM else "model",
                    "gamma": arm.gamma,
                    "fallback": None if arm.fallback is None else "model",
                }
                for arm in method.arms
            ]
            extra["bandit"] = {
                "selector": method.selector,
                "window_size": window,
                "seed": seed if method.selector == "thompson" else None,
                "state": "carried across this block's prompts on one engine",
            }
            extra["arm_report"] = arm_report(stats.arm_steps, method.arms, [False, True], window=50)
            extra["step_profile"] = summarise_steps(stats.steps)
            extra["host_syncs"] = probes[name]
            extra["compile"] = {
                "requested": engine.compile_decision.requested,
                "effective": engine.compiled,
                "decision": engine.compile_decision.reason,
                "fallback_reason": getattr(
                    method.group.target._inc, "compile_fallback_reason", None
                ),
                "fixed_shapes": method.fixed_shapes,
            }
            report = graph_report(_CAPTURES.report(f"{block_id}:{name}"), engine)
            report["group"] = method.group.name
            report["group_methods"] = list(method.group.methods)
            report["group_captures_in_process"] = sum(
                count
                for section, count in _CAPTURES.by_section.items()
                if section.rsplit(":", 1)[-1] in method.group.methods
            )
            extra["cuda_graphs"] = report
        elif name.endswith("_compiled"):
            extra["compile"] = {"requested": True, "effective": device.type == "cuda"}
            extra["cuda_graphs"] = _CAPTURES.report(f"{block_id}:{name}")
        if greedy and name not in ("hf_greedy",):
            for ref, key in (
                ("hf_greedy", "greedy_token_match"),
                ("hf_greedy_compiled", "greedy_token_match_compiled"),
            ):
                if name == ref:
                    continue
                found = match(stats, ref)
                if found is not None:
                    extra[key] = found
        notes = common_notes
        if is_spec and not spec_methods[name].fixed_shapes:
            notes += " compile_fixed_shapes=False: the unpadded 0.2.0 call shapes."
        # No vanilla autoregressive run here; HuggingFace has no acceptance.
        unmeasured = ["speedup_vs_ar_mean", "speedup_vs_ar_std"]
        if not is_spec:
            unmeasured += ["acceptance_rate_mean", "acceptance_rate_std"]
        if device.type != "cuda":
            unmeasured.append("gpu_memory_gb")
        documents[name] = assemble_result(
            model=model_name,
            method=name,
            gamma=max(arm.gamma for arm in spec_methods[name].arms) if is_spec else 0,
            dataset=dataset,
            n_samples=len(prompts),
            metrics=metrics,
            measurement_kind="toy_random_init_llama" if toy else "end_to_end",
            notes=notes,
            unmeasured=unmeasured,
            hardware=hardware,
            prompt_ids=prompt_ids,
            extra=extra,
        )
    return documents


def _forward_counts(stats: _Runs, is_spec: bool) -> dict[str, object]:
    """Target forwards, derived (no hooks on compiled forwards): prefill + one per step or token."""
    n_runs = len(stats.tokens)
    emitted = sum(stats.tokens)
    per_run = sum(len(run) for run in stats.arm_steps) if is_spec else emitted - n_runs
    target = n_runs + per_run
    counts: dict[str, object] = {
        "n_generations": n_runs,
        "tokens_emitted": emitted,
        "target_forwards": target,
        "target_forwards_per_emitted_token": target / emitted if emitted else 0.0,
        "source": "derived: a hook would force the compiled forward to recompile",
    }
    if is_spec:
        n_steps = per_run
        counts["n_steps"] = n_steps
        counts["target_forwards_per_step"] = 1.0 if n_steps else 0.0
        counts["emitted_tokens_per_step"] = emitted / n_steps if n_steps else 0.0
    return counts


def _note_cold(
    cold_start: dict[str, dict[str, object]],
    name: str,
    elapsed_ms: float,
    out: Any,  # noqa: ANN401
    model_name: str,
    config: dict[str, Any],
    *,
    warmup: bool,
) -> None:
    if name in cold_start:
        return
    key = (name, model_name, str(config.get("drafter", model_name)))
    tokens = getattr(out, "n_tokens_generated", None)
    if tokens is None:
        tokens = int(out.shape[1])
    cold_start[name] = {
        "first_call_ms": elapsed_ms,
        "first_call_new_tokens": int(tokens),
        "first_call_was_warmup": warmup,
        "first_in_process": key not in _COLD_STARTED,
        "source": "wall time of this method's first generate in this block (synchronized); "
        "a FlashSpec method's first call in a fresh wrapper group includes torch.compile "
        "and CUDA-graph capture",
    }
    _COLD_STARTED.add(key)
