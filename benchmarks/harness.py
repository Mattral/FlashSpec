"""Reproducible benchmark helpers.

Every result dict written by this module sets ``status`` to ``"measured"``
or is not written at all. Fields that were not measured are omitted and
listed under ``unmeasured``. Hardware is detected at runtime. A config
that asks for CUDA when CUDA is absent raises, and no JSON is written.

Medusa and EAGLE are not implemented in this repository. Callers must
not invent numbers for them.

Besides ``flashspec_ucb`` and ``vanilla_ar``, a config may list the
HuggingFace baselines ``hf_greedy`` (``generate(do_sample=False)``),
``hf_sample`` (``generate(do_sample=True, temperature=T)``) and
``hf_assisted`` (greedy ``generate`` with ``assistant_model`` set to the
drafter, using HuggingFace's own default assistant schedule). They need
HuggingFace modules, so ``model: toy`` skips them and ``model: toy_llama``
or a hub checkpoint runs them. All methods share the target, prompts,
dtype and ``max_new_tokens``. With ``temperatures: [...]`` each dataset
runs once per mode, and only the matching plain baseline runs
(``hf_greedy`` at 0, ``hf_sample`` otherwise); ``hf_assisted`` follows
the mode. Loaded checkpoints get a neutral ``GenerationConfig()`` so the
baselines are plain greedy or plain temperature sampling, not the
checkpoint's shipped top-k/top-p/repetition-penalty defaults.

Every document carries ``forward_counts``, ``host_syncs`` and
``nonfinite_logits`` (the last two from one extra untimed generation per
method on the first prompt), ``decoding``, ``drafter``,
``prompt_format`` and ``prompt_tokens``. FlashSpec also carries
``step_profile``; greedy FlashSpec and ``hf_assisted`` carry
``greedy_token_match``.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from benchmarks.prompt_sets import (
    ALPACA_REASON,
    REAL_DATASETS,
    canonical_dataset,
    load_prompt_set,
    tokenize_prompts,
)
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.drafter import load_drafters, unregister
from flashspec.engine.kv_cache import (
    CacheUnavailableError,
    KvSession,
    model_accepts_past_key_values,
)
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.instrument import ForwardRecorder, NonFiniteLogits, count_host_syncs
from flashspec.metrics.step_profile import StepRecord, summarise_steps
from flashspec.utils.config import BanditConfig, FlashSpecConfig, MetricsConfig, SamplingConfig
from flashspec.utils.cpu import cpu_has_avx2
from flashspec.utils.device import set_seed

__all__ = [
    "DeviceUnavailableError",
    "assemble_result",
    "autoregressive_generate",
    "detect_hardware",
    "git_commit",
    "load_benchmark_config",
    "percentile",
    "resolve_device",
    "run_paired_generations",
    "summarise_runs",
]


#: Baselines that call HuggingFace ``generate``. Each needs a module with
#: ``generate``; ``hf_assisted`` also needs a HuggingFace drafter.
HF_BASELINES: tuple[str, ...] = ("hf_greedy", "hf_sample", "hf_assisted")

#: ``model`` values that build randomly initialised in-process networks.
TOY_MODELS: frozenset[str] = frozenset({"toy", "toy_llama"})


class DeviceUnavailableError(RuntimeError):
    """The config asked for a device this machine does not have."""


class _TinyCausalLM(nn.Module):
    """Small causal LM used when a config sets ``model: toy``.

    This is not a language model checkpoint. Results that use it must
    say so in ``measurement_kind``.
    """

    def __init__(self, vocab_size: int, hidden: int, n_layers: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4 if hidden % 4 == 0 else 1,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return a namespace with ``logits`` shaped ``(batch, seq, vocab)``."""
        hidden = self.transformer(self.embed(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def git_commit() -> str:
    """Return the short git hash, or ``unknown`` when git is unavailable.

    Returns
    -------
    str
        Short revision, or ``unknown``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = result.stdout.strip()
    return text or "unknown"


def detect_hardware() -> dict[str, object]:
    """Describe the environment that is about to be measured.

    The OS name is `platform.system()` only. The CPU identifier and the
    OS build number are not recorded. GPU keys are omitted when CUDA is
    not available. Nothing in the returned dict is a design target.

    Returns
    -------
    dict[str, object]
        OS name, architecture, Python, PyTorch, AVX2, and CUDA fields
        when present.
    """
    info: dict[str, object] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "avx2": cpu_has_avx2(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        cuda = torch.version.cuda
        if cuda:
            info["cuda"] = cuda
        info["gpu_count"] = torch.cuda.device_count()
    return info


def resolve_device(requested: str) -> torch.device:
    """Resolve a config device string.

    Parameters
    ----------
    requested : str
        ``auto``, ``cpu``, or a torch device string such as ``cuda:0``.

    Returns
    -------
    torch.device
        The device generations will run on.

    Raises
    ------
    DeviceUnavailableError
        If the config names a CUDA device and CUDA is not available.
        Callers must not write a result file in that case.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError(
            f"Config requests {requested}, but CUDA is not available on this "
            "machine. No result file was written. Use device: auto or device: cpu "
            "to measure here, or run this config on a machine that has that GPU."
        )
    return device


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile.

    Parameters
    ----------
    values : list[float]
        Sample values. Must be non-empty.
    pct : float
        Percent in ``[0, 100]``.

    Returns
    -------
    float
        The nearest-rank value.

    Raises
    ------
    ValueError
        If ``values`` is empty or ``pct`` is outside ``[0, 100]``.
    """
    if not values:
        raise ValueError("percentile requires at least one value.")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100]; got {pct}.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[rank]


def summarise_runs(
    latencies_ms: list[float],
    token_counts: list[int],
    acceptance_rates: list[float] | None = None,
) -> dict[str, float]:
    """Aggregate per-generation samples into the benchmark metric dict.

    Parameters
    ----------
    latencies_ms : list[float]
        Wall time of each generation, in milliseconds.
    token_counts : list[int]
        New tokens produced by each generation.
    acceptance_rates : list[float] or None
        Per-generation acceptance rates. ``None`` means acceptance was
        not measured (vanilla autoregressive has no draft tokens).

    Returns
    -------
    dict[str, float]
        Measured metric keys only. Acceptance keys are absent when
        ``acceptance_rates`` is ``None``.

    Raises
    ------
    ValueError
        If the latency and token-count lists differ in length or are empty.
    """
    if not latencies_ms or len(latencies_ms) != len(token_counts):
        raise ValueError("latencies_ms and token_counts must be non-empty and aligned.")
    throughputs = [
        (tokens / (latency / 1000.0)) if latency > 0.0 else 0.0
        for tokens, latency in zip(token_counts, latencies_ms, strict=True)
    ]
    mean_tps = sum(throughputs) / len(throughputs)
    var_tps = sum((item - mean_tps) ** 2 for item in throughputs) / len(throughputs)
    ordered = sorted(throughputs)
    if len(ordered) > 1:
        # Linear interpolation between order statistics (statistics.quantiles,
        # method="inclusive"), so the IQR uses every prompt.
        p25, _median, p75 = statistics.quantiles(ordered, n=4, method="inclusive")
    else:
        p25 = p75 = ordered[0]
    metrics: dict[str, float] = {
        "tokens_per_second_mean": mean_tps,
        "tokens_per_second_std": var_tps**0.5,
        "tokens_per_second_median": statistics.median(ordered),
        "tokens_per_second_p25": p25,
        "tokens_per_second_p75": p75,
        "tokens_per_second_iqr": p75 - p25,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "p99_latency_ms": percentile(latencies_ms, 99),
    }
    if torch.cuda.is_available():
        metrics["gpu_memory_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    if acceptance_rates is not None:
        if len(acceptance_rates) != len(latencies_ms):
            raise ValueError("acceptance_rates must align with latencies_ms.")
        mean_alpha = sum(acceptance_rates) / len(acceptance_rates)
        var_alpha = sum((item - mean_alpha) ** 2 for item in acceptance_rates) / len(
            acceptance_rates
        )
        metrics["acceptance_rate_mean"] = mean_alpha
        metrics["acceptance_rate_std"] = var_alpha**0.5
    return metrics


def assemble_result(
    *,
    model: str,
    method: str,
    gamma: int,
    dataset: str,
    n_samples: int,
    metrics: dict[str, float],
    measurement_kind: str,
    notes: str,
    unmeasured: list[str],
    hardware: dict[str, object] | None = None,
    prompt_ids: list[str] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a result document. Unmeasured metrics are not given fake zeros.

    Parameters
    ----------
    model : str
        Model id, or ``toy`` for the in-process network.
    method : str
        Method name, for example ``flashspec_ucb`` or ``vanilla_ar``.
    gamma : int
        Speculation length. Vanilla AR records the gamma it was compared
        against; it does not itself speculate.
    dataset : str
        Dataset or prompt-source name.
    n_samples : int
        Number of timed generations.
    metrics : dict[str, float]
        Measured numbers only.
    measurement_kind : str
        Short label such as ``end_to_end`` or ``toy_random_init``.
    notes : str
        Human-readable scope of the measurement.
    unmeasured : list[str]
        Metric names deliberately omitted.
    hardware : dict or None
        Override for tests. The default calls :func:`detect_hardware`.
    prompt_ids : list[str] or None
        Ids from a fixed prompt set. Omitted for synthetic token ids.
    extra : dict or None
        Additional measured sections, such as ``step_profile``. Keys
        that collide with the fixed fields raise ``ValueError``.

    Returns
    -------
    dict[str, object]
        JSON-ready document with ``status`` = ``measured``.
    """
    document: dict[str, object] = {
        "status": "measured",
        "measurement_kind": measurement_kind,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "hardware": hardware if hardware is not None else detect_hardware(),
        "model": model,
        "method": method,
        "gamma": gamma,
        "dataset": dataset,
        "n_samples": n_samples,
        "metrics": metrics,
        "unmeasured": unmeasured,
        "notes": notes,
    }
    if prompt_ids is not None:
        document["prompt_ids"] = list(prompt_ids)
    if extra:
        clash = set(extra) & set(document)
        if clash:
            raise ValueError(f"extra sections overwrite fixed fields: {sorted(clash)}")
        document.update(extra)
    return document


def write_result(path: Path, result: dict[str, object]) -> None:
    """Write ``result`` as JSON.

    Parameters
    ----------
    path : Path
        Destination file. Parent directories are created.
    result : dict
        Document produced by :func:`assemble_result`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def load_benchmark_config(path: Path) -> dict[str, Any]:
    """Load a benchmark YAML file.

    Uses PyYAML when it is installed. Otherwise a small subset parser
    handles the flat configs in ``benchmarks/configs``.

    Parameters
    ----------
    path : Path
        YAML file.

    Returns
    -------
    dict
        Config mapping.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a mapping.")
    return loaded


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the flat YAML used by the checked-in benchmark configs.

    Parameters
    ----------
    text : str
        File contents.

    Returns
    -------
    dict
        Nested mapping. Lists of scalars are supported. Nested mappings
        of scalars are supported one level deep.
    """
    root: dict[str, Any] = {}
    current_list: str | None = None
    current_map: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith(" ") or raw.startswith("\t"):
            stripped = raw.strip()
            if current_list is not None and stripped.startswith("- "):
                root[current_list].append(_coerce(stripped[2:].strip()))
                continue
            if current_map is not None and ":" in stripped:
                key, value = stripped.split(":", 1)
                root[current_map][key.strip()] = _coerce(value.strip())
                continue
        current_list = None
        current_map = None
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            # The next indented block decides list vs map. Lists are the
            # common case in these configs; a later nested key flips it.
            root[key] = []
            current_list = key
            continue
        root[key] = _coerce(value)
    # Promote empty-then-mapping blocks. The parser above stores a list
    # until it sees a nested ``key:`` that is not a dash, but those lines
    # are handled only when current_map is set. Re-scan for that pattern.
    return _repair_nested_maps(text, root)


def _repair_nested_maps(text: str, root: dict[str, Any]) -> dict[str, Any]:
    """Turn list placeholders into dicts when the block is a mapping.

    Parameters
    ----------
    text : str
        Original YAML text.
    root : dict
        First-pass parse.

    Returns
    -------
    dict
        Parse with one-level nested maps restored.
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        if raw and not raw.startswith((" ", "\t", "#")) and raw.endswith(":"):
            key = raw[:-1].strip()
            block: dict[str, Any] = {}
            is_map = False
            index += 1
            while index < len(lines) and (
                lines[index].startswith((" ", "\t")) or not lines[index].strip()
            ):
                stripped = lines[index].strip()
                index += 1
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("- "):
                    is_map = False
                    break
                if ":" in stripped:
                    is_map = True
                    child_key, child_value = stripped.split(":", 1)
                    block[child_key.strip()] = _coerce(child_value.strip())
            if is_map:
                root[key] = block
            continue
        index += 1
    return root


def _coerce(value: str) -> bool | int | float | str | None:
    """Convert a scalar YAML token into bool, int, float, str, or None.

    Parameters
    ----------
    value : str
        Raw token, without comments.

    Returns
    -------
    bool or int or float or str or None
        Coerced value. ``null`` becomes ``None``.
    """
    token = value.split("#", 1)[0].strip().strip('"').strip("'")
    if token in {"true", "True"}:
        return True
    if token in {"false", "False"}:
        return False
    if token in {"null", "Null", "~"}:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _uses_hf_generate(model: nn.Module) -> bool:
    """Return whether ``model`` is a HuggingFace generator with a KV cache."""
    generate = getattr(model, "generate", None)
    prepare = getattr(model, "prepare_inputs_for_generation", None)
    return callable(generate) and callable(prepare)


def _hf_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    *,
    do_sample: bool = True,
    assistant_model: nn.Module | None = None,
    eos_token_ids: list[int] | None = None,
) -> torch.Tensor:
    """Decode with ``model.generate`` and the model's own KV cache.

    Parameters
    ----------
    model : nn.Module
        HuggingFace causal LM.
    input_ids : torch.Tensor
        Prompt. Shape ``(batch, seq)``.
    max_new_tokens : int
        Number of new tokens. The call does not stop early on EOS.
    temperature : float
        Sampling temperature passed through to ``generate``. Ignored when
        ``do_sample`` is false.
    do_sample : bool
        False decodes greedily.
    assistant_model : nn.Module or None
        Drafter for HuggingFace assisted generation. Its own generation
        config sets the assistant schedule.
    eos_token_ids : list of int or None
        Stop at these ids. ``None`` (the default) generates exactly
        ``max_new_tokens`` with no EOS handling at all.

    Returns
    -------
    torch.Tensor
        New token ids. Shape ``(batch, max_new_tokens)``.

    Raises
    ------
    RuntimeError
        If ``generate`` does not return ``max_new_tokens`` new tokens.
    """
    prompt_len = int(input_ids.shape[1])
    pad = getattr(getattr(model, "config", None), "pad_token_id", None)
    kwargs: dict[str, Any] = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if eos_token_ids:
        kwargs["eos_token_id"] = list(eos_token_ids)
    else:
        kwargs.update({"min_new_tokens": max_new_tokens, "eos_token_id": None})
    if do_sample:
        # Plain temperature sampling, as FlashSpec does. GenerationConfig()
        # resolves top_k to 50 at generate time; 0 disables top-k.
        kwargs.update({"temperature": temperature, "top_k": 0, "top_p": 1.0})
    else:
        # Clear sampling defaults a checkpoint may ship, so generate does
        # not warn that they are ignored.
        kwargs.update({"temperature": None, "top_p": None, "top_k": None})
    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
    if pad is not None:
        kwargs["pad_token_id"] = pad
    with torch.no_grad():
        output = model.generate(input_ids, **kwargs)
    new_tokens = output[:, prompt_len:]
    if eos_token_ids:
        return new_tokens
    if int(new_tokens.shape[1]) != max_new_tokens:
        raise RuntimeError(
            f"generate returned {int(new_tokens.shape[1])} new tokens, expected {max_new_tokens}."
        )
    return new_tokens


def _cached_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> torch.Tensor:
    """Sample one token at a time from ``past_key_values``.

    Parameters
    ----------
    model : nn.Module
        Module whose forward accepts ``past_key_values``.
    input_ids : torch.Tensor
        One prompt. Shape ``(1, seq)``.
    max_new_tokens : int
        Number of new tokens.
    temperature : float
        Applied to logits before softmax.

    Returns
    -------
    torch.Tensor
        New token ids. Shape ``(1, max_new_tokens)``.
    """
    session = KvSession(model)
    logits = session.sync(input_ids)
    pieces: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        next_id = _next_token(logits, temperature)
        pieces.append(next_id)
        logits = session.append(next_id)[:, -1, :]
    return torch.cat(pieces, dim=1)


def _next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample from ``logits / temperature``, or take the argmax at temperature 0."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits if temperature == 1.0 else logits / temperature
    return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)


def autoregressive_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Autoregressive sample used as the plain-decoding baseline.

    A HuggingFace causal LM goes through ``generate(..., use_cache=True)``,
    which is the cached baseline. Any other batch-1 module that accepts
    ``past_key_values`` uses that cache directly. Everything else forwards
    the growing context on every token.

    Parameters
    ----------
    model : nn.Module
        Module whose forward returns ``.logits``.
    input_ids : torch.Tensor
        Prompt. Shape ``(batch, seq)``.
    max_new_tokens : int
        Number of new tokens.
    temperature : float
        Applied to logits before softmax. ``0`` decodes greedily.

    Returns
    -------
    torch.Tensor
        New token ids. Shape ``(batch, max_new_tokens)``.
    """
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0; got {temperature}.")
    if _uses_hf_generate(model):
        return _hf_generate(
            model, input_ids, max_new_tokens, temperature, do_sample=temperature > 0.0
        )
    if input_ids.shape[0] == 1 and model_accepts_past_key_values(model):
        try:
            return _cached_generate(model, input_ids, max_new_tokens, temperature)
        except CacheUnavailableError:
            pass
    context = input_ids
    pieces: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(context).logits[:, -1, :].float()
        next_id = _next_token(logits, temperature)
        pieces.append(next_id)
        context = torch.cat([context, next_id], dim=1)
    return torch.cat(pieces, dim=1)


def _torch_dtype_from_config(config: dict[str, Any]) -> torch.dtype:
    """Map a config ``dtype`` string to a torch dtype.

    Parameters
    ----------
    config : dict
        Benchmark config. Missing ``dtype`` means float32.

    Returns
    -------
    torch.dtype
        ``float32``, ``float16``, or ``bfloat16``.

    Raises
    ------
    RuntimeError
        If the name is not one of those three. Callers must not write a
        result file after this raise.
    """
    raw = str(config.get("dtype", "float32")).lower().strip()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[raw]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported dtype {raw!r}. Use float32, float16, or bfloat16. "
            "No result file was written."
        ) from exc


def _assert_shared_vocab(drafter: nn.Module, target: nn.Module) -> None:
    """Refuse a drafter and target that do not share token ids.

    Parameters
    ----------
    drafter : nn.Module
        Draft model.
    target : nn.Module
        Target model.

    Raises
    ------
    RuntimeError
        If the vocabulary sizes differ. Callers must not write a result.
    """
    draft_vocab = _module_vocab(drafter, {})
    target_vocab = _module_vocab(target, {})
    if draft_vocab != target_vocab:
        raise RuntimeError(
            f"Drafter vocab {draft_vocab} and target vocab {target_vocab} differ. "
            "Speculative decoding would compare different token ids. "
            "No result file was written."
        )


def _module_vocab(module: nn.Module, config: dict[str, Any]) -> int:
    """Read a vocabulary size from a module, its config, or the YAML file.

    Parameters
    ----------
    module : nn.Module
        Target module.
    config : dict
        Benchmark config, used only as a fallback.

    Returns
    -------
    int
        Vocabulary size used to sample synthetic prompts.
    """
    direct = getattr(module, "vocab_size", None)
    if isinstance(direct, int):
        return direct
    model_config = getattr(module, "config", None)
    size = getattr(model_config, "vocab_size", None) if model_config is not None else None
    if isinstance(size, int):
        return size
    return int(config.get("vocab_size", 64))


def _build_toy_modules(config: dict[str, Any], device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Create a drafter and a target from a ``model: toy`` config.

    Parameters
    ----------
    config : dict
        Benchmark config. Missing size keys use small defaults.
    device : torch.device
        Device both modules are moved to.

    Returns
    -------
    tuple[nn.Module, nn.Module]
        ``(drafter, target)``.
    """
    vocab = int(config.get("vocab_size", 64))
    target = (
        _TinyCausalLM(
            vocab,
            int(config.get("target_hidden", 64)),
            int(config.get("target_layers", 2)),
        )
        .to(device)
        .eval()
    )
    drafter = (
        _TinyCausalLM(
            vocab,
            int(config.get("drafter_hidden", 32)),
            int(config.get("drafter_layers", 1)),
        )
        .to(device)
        .eval()
    )
    return drafter, target


def _build_toy_llama_modules(
    config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, nn.Module]:
    """Create a random tiny Llama drafter and target from a ``model: toy_llama`` config.

    Unlike ``model: toy``, these are HuggingFace modules with a real
    ``DynamicCache`` and ``generate``, so the KV-cache paths and the
    HuggingFace baselines run on them.

    Parameters
    ----------
    config : dict
        Benchmark config. ``vocab_size``, ``target_hidden``,
        ``target_layers``, ``drafter_hidden``, ``drafter_layers`` and
        ``initializer_range`` are read, with small defaults.
    device : torch.device
        Device both modules are moved to.

    Returns
    -------
    tuple[nn.Module, nn.Module]
        ``(drafter, target)``.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    vocab = int(config.get("vocab_size", 512))
    init_range = float(config.get("initializer_range", 0.1))
    # Real prompt sets have no prompt_length cap; leave room for them.
    prompt_room = config.get("prompt_length") or 2048
    max_positions = int(prompt_room) + int(config.get("max_new_tokens", 8)) + 64

    def _one(hidden: int, layers: int) -> nn.Module:
        heads = 4 if hidden % 4 == 0 else 1
        llama_config = LlamaConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=hidden * 2,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            max_position_embeddings=max_positions,
            pad_token_id=0,
            bos_token_id=None,
            eos_token_id=None,
            initializer_range=init_range,
        )
        return LlamaForCausalLM(llama_config).to(device).eval()

    target = _one(int(config.get("target_hidden", 64)), int(config.get("target_layers", 4)))
    drafter = _one(int(config.get("drafter_hidden", 32)), int(config.get("drafter_layers", 2)))
    return drafter, target


def _prompts(
    config: dict[str, Any],
    device: torch.device,
    tokenizer: object | None = None,
) -> tuple[list[torch.Tensor], list[str] | None, str]:
    """Build prompts for one config.

    Parameters
    ----------
    config : dict
        Benchmark config.
    device : torch.device
        Destination device.
    tokenizer : object or None
        Target tokenizer. Required when ``dataset`` is a fixed prompt set.

    Returns
    -------
    tuple
        Prompt tensors, prompt ids or ``None`` for synthetic ids, and a
        note fragment. The fragment is empty for synthetic prompts.

    Raises
    ------
    RuntimeError
        If a real prompt set has no tokenizer or an empty encoding.
    """
    slug = canonical_dataset(str(config.get("dataset", "synthetic_prompts")))
    if slug in REAL_DATASETS:
        return _real_prompts(config, device, tokenizer, slug)
    n_samples = int(config.get("n_samples", config.get("n_measurement_steps", 4)))
    vocab = int(config.get("vocab_size", 64))
    prompt_length = int(config.get("prompt_length", 8))
    seed = int(config.get("seed", 0))
    prompts: list[torch.Tensor] = []
    for index in range(n_samples):
        generator = torch.Generator().manual_seed(seed + index)
        prompts.append(torch.randint(0, vocab, (1, prompt_length), generator=generator).to(device))
    return prompts, None, ""


def _real_prompts(
    config: dict[str, Any],
    device: torch.device,
    tokenizer: object | None,
    slug: str,
) -> tuple[list[torch.Tensor], list[str], str]:
    """Tokenize a fixed prompt set and describe which ids were used.

    Parameters
    ----------
    config : dict
        Benchmark config. ``n_samples`` or ``n_measurement_steps`` slices
        the ordered set. A request larger than the set uses each prompt
        once.
    device : torch.device
        Destination device.
    tokenizer : object or None
        Target tokenizer.
    slug : str
        Canonical dataset name.

    Returns
    -------
    tuple
        Tensors, ids, and a note fragment that starts with a space.

    Raises
    ------
    RuntimeError
        If ``tokenizer`` is missing.
    """
    if tokenizer is None:
        raise RuntimeError("A real prompt set requires a tokenizer. No result file was written.")
    records = load_prompt_set(slug)
    if "n_samples" in config:
        requested = int(config["n_samples"])
        source = "n_samples"
    elif "n_measurement_steps" in config:
        requested = int(config["n_measurement_steps"])
        source = "n_measurement_steps"
    else:
        requested = 4
        source = "n_samples"
    capped = requested > len(records)
    chosen = records if capped else records[:requested]
    raw_limit = config.get("prompt_length")
    limit = int(raw_limit) if raw_limit is not None else None
    chat = bool(config.get("chat_template", False))
    if chat:
        prompts = [
            torch.tensor(
                [_chat_ids(tokenizer, item["text"])[:limit]], dtype=torch.long, device=device
            )
            for item in chosen
        ]
    else:
        prompts = tokenize_prompts(chosen, tokenizer, device, limit)
    ids = [item["id"] for item in chosen]
    how = (
        "as one user turn through the target's chat template (generation prompt added)"
        if chat
        else "with the target tokenizer"
    )
    note = f" Prompts are the fixed {slug} set ({', '.join(ids)}), tokenized {how}."
    if capped:
        note += (
            f" {source} asked for {requested} prompts; the fixed set has "
            f"{len(records)}, so each prompt is used once."
        )
    if str(config.get("model", "toy")) in TOY_MODELS:
        note += (
            " The network is randomly initialised, so acceptance is not a checkpoint measurement."
        )
    return prompts, ids, note


def _chat_ids(tokenizer: object, text: str) -> list[int]:
    """Token ids of ``text`` as one user turn, through the tokenizer's chat template.

    Raises
    ------
    RuntimeError
        If the tokenizer has no chat template or returns no ids.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply) or getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "chat_template is set but the target tokenizer has no chat template. "
            "No result file was written."
        )
    encoded = apply([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    raw = encoded["input_ids"] if not isinstance(encoded, list) else encoded
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    ids = [int(token) for token in raw]
    if not ids:
        raise RuntimeError("The chat template produced no tokens. No result file was written.")
    return ids


def run_paired_generations(
    config: dict[str, Any],
    *,
    drafter_module: nn.Module | None = None,
    target_module: nn.Module | None = None,
) -> dict[str, dict[str, object]]:
    """Time FlashSpec UCB against vanilla autoregressive decoding.

    Parameters
    ----------
    config : dict
        Benchmark config. ``device`` may be ``auto``.
    drafter_module : nn.Module or None
        Injected drafter. Required for ``model: toy`` unless the config
        can build one. Hub ids are loaded when no module is injected and
        the model is not ``toy``.
    target_module : nn.Module or None
        Injected target. Same rules as ``drafter_module``.

    Returns
    -------
    dict[str, dict]
        Keys ``flashspec_ucb`` and ``vanilla_ar``. FlashSpec includes
        ``speedup_vs_ar_mean`` when both throughputs were measured.
        The standard deviation of the speedup is listed as unmeasured
        unless more than one paired ratio is available, in which case it
        is computed across paired samples.

    Raises
    ------
    DeviceUnavailableError
        If the requested CUDA device is missing.
    RuntimeError
        If a non-toy model cannot be loaded, Alpaca is requested, or a
        real prompt set has no tokenizer. No partial result is implied.
    """
    device = resolve_device(str(config.get("device", "auto")))
    model_name = str(config.get("model", "toy"))
    gamma = int(config.get("gamma", 4))
    max_new_tokens = int(config.get("max_new_tokens", 8))
    temperature = float(config.get("temperature", 1.0))
    n_warmup = int(config.get("n_warmup", config.get("n_warmup_steps", 1)))
    dataset = canonical_dataset(str(config.get("dataset", "synthetic_prompts")))
    tokenizer = config.get("tokenizer")
    toy = model_name in TOY_MODELS
    if dataset == "alpaca":
        raise RuntimeError(ALPACA_REASON)
    if dataset in REAL_DATASETS and tokenizer is None and toy:
        raise RuntimeError("A real prompt set requires a tokenizer. No result file was written.")
    seed = int(config.get("seed", 0))
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if drafter_module is None or target_module is None:
        if model_name == "toy_llama":
            drafter_module, target_module = _build_toy_llama_modules(config, device)
        elif toy:
            drafter_module, target_module = _build_toy_modules(config, device)
        else:
            drafter_module, target_module = _load_checkpoint_modules(config, device)
    if drafter_module is None or target_module is None:
        raise RuntimeError("Internal error: drafter and target were not loaded.")
    if dataset in REAL_DATASETS and tokenizer is None:
        tokenizer = _load_tokenizer(model_name)
    _assert_shared_vocab(drafter_module, target_module)
    config = {**config, "vocab_size": _module_vocab(target_module, config), "dataset": dataset}
    weight_dtype = next(target_module.parameters()).dtype

    drafter_name = "bench-drafter"
    unregister(drafter_name)
    try:
        return _run_loaded_pair(
            config=config,
            device=device,
            drafter_module=drafter_module,
            target_module=target_module,
            drafter_name=drafter_name,
            model_name=model_name,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_warmup=n_warmup,
            dataset=dataset,
            seed=seed,
            toy=toy,
            weight_dtype=weight_dtype,
            tokenizer=tokenizer,
        )
    finally:
        unregister(drafter_name)


def _load_checkpoint_modules(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    """Load a target and a drafter from the HuggingFace hub or a local path.

    Parameters
    ----------
    config : dict
        Must contain ``model`` and ``drafter``.
    device : torch.device
        Device both modules are moved to.

    Returns
    -------
    tuple[nn.Module, nn.Module]
        ``(drafter, target)``.

    Raises
    ------
    RuntimeError
        If the weights cannot be loaded. The caller must not write a result.
    """
    from transformers import AutoModelForCausalLM

    target_name = str(config.get("model", ""))
    drafter_name = config.get("drafter")
    if not isinstance(drafter_name, str) or not drafter_name:
        raise RuntimeError(
            "Config is missing a string 'drafter' id. Refusing to guess one "
            "or to write empty metrics."
        )
    weight_dtype = _torch_dtype_from_config(config)
    try:
        target = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=weight_dtype)
        drafter = AutoModelForCausalLM.from_pretrained(drafter_name, torch_dtype=weight_dtype)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load target {target_name!r} and drafter {drafter_name!r}: {exc}. "
            "No result file was written."
        ) from exc
    target.eval()
    drafter.eval()
    _assert_shared_tokenizer(target_name, drafter_name)
    for module in (target, drafter):
        _neutralise_generation_config(module)
    return drafter.to(device), target.to(device)


def _neutralise_generation_config(model: nn.Module) -> dict[str, object]:
    """Replace a checkpoint's shipped generation defaults with HuggingFace's neutral ones.

    Checkpoints ship sampling defaults (Qwen2.5-Instruct: ``top_k=20``,
    ``top_p=0.8``, ``repetition_penalty=1.1``). ``generate`` applies them
    unless every one is overridden, and a repetition penalty applies even
    with ``do_sample=False``. The baselines must be plain greedy and plain
    temperature sampling, as FlashSpec is, so the shipped config is
    replaced by ``GenerationConfig()`` with only ``pad_token_id`` kept.

    Parameters
    ----------
    model : nn.Module
        HuggingFace model. Modules without ``generation_config`` are left alone.

    Returns
    -------
    dict
        The checkpoint's shipped non-default settings, also stored on the
        module as ``flashspec_generation_overrides``. All of them are
        replaced except ``pad_token_id``. The shipped ``eos_token_id`` is
        kept separately as ``flashspec_eos_token_ids`` (used only when a
        config sets ``stop_at_eos``).
    """
    from transformers import GenerationConfig

    shipped = getattr(model, "generation_config", None)
    if shipped is None:
        return {}
    shipped_settings = {
        key: value
        for key, value in shipped.to_diff_dict().items()
        if key not in {"transformers_version", "_from_model_config"}
    }
    eos = shipped.eos_token_id
    eos_ids = [] if eos is None else ([int(eos)] if isinstance(eos, int) else [int(e) for e in eos])
    model.generation_config = GenerationConfig(pad_token_id=shipped.pad_token_id)
    model.flashspec_generation_overrides = shipped_settings  # type: ignore[assignment]
    model.flashspec_eos_token_ids = eos_ids  # type: ignore[assignment]
    return shipped_settings


def _assert_shared_tokenizer(target_name: str, drafter_name: str) -> None:
    """Refuse a pair whose tokenizers map text to different ids.

    Equal ``vocab_size`` is not enough: two tokenizers can share a size
    and still assign ids differently.

    Raises
    ------
    RuntimeError
        If the vocabularies (including added tokens) differ. No result
        file is written.
    """
    target_tok = _load_tokenizer(target_name)
    drafter_tok = _load_tokenizer(drafter_name)
    target_vocab = target_tok.get_vocab()  # type: ignore[attr-defined]
    drafter_vocab = drafter_tok.get_vocab()  # type: ignore[attr-defined]
    if target_vocab != drafter_vocab:
        differing = sorted(
            key
            for key in set(target_vocab) | set(drafter_vocab)
            if target_vocab.get(key) != drafter_vocab.get(key)
        )
        raise RuntimeError(
            f"Tokenizers of {target_name!r} and {drafter_name!r} differ on "
            f"{len(differing)} entries (first: {differing[:5]}). Speculative decoding "
            "would compare different token ids. No result file was written."
        )


def _load_tokenizer(model_name: str) -> object:
    """Load a tokenizer for ``model_name`` before any result is written.

    Parameters
    ----------
    model_name : str
        Target checkpoint id or local path.

    Returns
    -------
    object
        Tokenizer from ``AutoTokenizer.from_pretrained``.

    Raises
    ------
    RuntimeError
        If the tokenizer cannot be loaded. The caller writes nothing.
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load a tokenizer for {model_name!r}: {exc}. No result file was written."
        ) from exc


_SPEC = "flashspec_ucb"


@dataclass
class _MethodRuns:
    """Measured generations of one method over the prompt set."""

    latency_ms: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    alpha: list[float] = field(default_factory=list)
    outputs: list[list[int]] = field(default_factory=list)
    target_forwards: int = 0
    drafter_forwards: int = 0
    peak_memory_bytes: int = 0
    steps: list[StepRecord] = field(default_factory=list)


def _run_loaded_pair(
    *,
    config: dict[str, Any],
    device: torch.device,
    drafter_module: nn.Module,
    target_module: nn.Module,
    drafter_name: str,
    model_name: str,
    gamma: int,
    max_new_tokens: int,
    temperature: float,
    n_warmup: int,
    dataset: str,
    seed: int,
    toy: bool,
    weight_dtype: torch.dtype,
    tokenizer: object | None = None,
) -> dict[str, dict[str, object]]:
    """Time one loaded drafter/target pair. See :func:`run_paired_generations`."""
    greedy = temperature == 0.0
    drafters = load_drafters(
        [drafter_name],
        device=device,
        dtype=weight_dtype,
        temperature=temperature,
        models={drafter_name: drafter_module},
    )
    target = TargetModel(model_name, device, weight_dtype, model=target_module)
    stop_at_eos = bool(config.get("stop_at_eos", False))
    eos_ids = list(getattr(target_module, "flashspec_eos_token_ids", []) or [])
    if stop_at_eos and not eos_ids:
        raise RuntimeError("stop_at_eos is set but the target has no eos_token_id.")
    engine_config = FlashSpecConfig(
        drafter_name=drafter_name,
        target_name=model_name,
        device=str(device),
        dtype=str(weight_dtype).removeprefix("torch."),
        max_new_tokens=max_new_tokens,
        eos_token_ids=tuple(eos_ids) if stop_at_eos else (),
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=temperature, seed=seed),
        metrics=MetricsConfig(profile_steps=True),
    )
    engine = SpeculativeEngine(
        engine_config,
        drafters,
        target,
        UCB1Selector(n_arms=1, window_size=0),
    )
    prompts, prompt_ids, prompt_note = _prompts(config, device, tokenizer)
    if model_name == "toy_llama":
        kind = "toy_random_init_llama"
    elif toy:
        kind = "toy_random_init_transformer"
    else:
        kind = "end_to_end"
    shared_notes = (
        "In-process randomly initialised transformer. Not a published "
        "checkpoint and not comparable to the Llama design targets."
        if toy
        else (
            "End-to-end generation on the loaded checkpoint versus a vanilla "
            "autoregressive loop on the same target. "
            f"Weights were loaded as {weight_dtype}."
        )
    )
    if not toy and "synthetic" in dataset:
        shared_notes += (
            " Prompts are synthetic token ids drawn from the shared vocabulary, "
            "not MT-Bench, HumanEval, or Alpaca."
        )
    if prompt_note:
        shared_notes += prompt_note
    if greedy:
        shared_notes += " Decoding is greedy (temperature 0)."
    else:
        shared_notes += f" Sampling at temperature {temperature}."
    if stop_at_eos:
        shared_notes += f" Every method stops at EOS {eos_ids}; max_new_tokens is a cap."
    else:
        shared_notes += (
            f" Every method generates exactly {max_new_tokens} tokens; EOS is neither "
            "a stop nor suppressed."
        )
    overrides = getattr(target_module, "flashspec_generation_overrides", None)
    if overrides:
        shared_notes += (
            " The checkpoint's shipped generation defaults were replaced by "
            "GenerationConfig() for the HuggingFace baselines (see generation_config_overrides)."
        )
    common_extra: dict[str, object] = {
        "decoding": {
            "mode": "greedy" if greedy else "sample",
            "temperature": temperature,
            "stop_at_eos": stop_at_eos,
            "max_new_tokens": max_new_tokens,
        },
        "drafter": str(config.get("drafter", model_name)),
        "prompt_format": "chat_template" if config.get("chat_template") else "raw",
        "prompt_tokens": [int(prompt.shape[1]) for prompt in prompts],
        "method_order": "rotated by one position per prompt",
    }
    if overrides is not None:
        common_extra["generation_config_overrides"] = {
            "replaced_with": "transformers GenerationConfig() defaults; pad_token_id kept",
            "shipped_target": overrides,
            "shipped_drafter": getattr(drafter_module, "flashspec_generation_overrides", {}),
        }
    spec_notes = shared_notes
    if len(drafters) < 2:
        spec_notes += (
            " One drafter is loaded, so the bandit cannot adapt; this row "
            "measures speculative decoding with a fixed drafter."
        )
    if _uses_hf_generate(target_module):
        ar_notes = (
            "Vanilla autoregressive baseline via HuggingFace generate() "
            "with the KV cache, on the same target and prompts. " + shared_notes
        )
    else:
        ar_notes = "Vanilla autoregressive loop on the same target and prompts. " + shared_notes

    requested = [str(name) for name in (config.get("methods") or [])]
    # vanilla_ar runs when listed, or when no methods are listed (older configs).
    run_ar = not requested or "vanilla_ar" in requested
    if run_ar and stop_at_eos:
        print("Skipping vanilla_ar: it has no EOS handling. No numbers were written for it.")
        run_ar = False
    runnable_hf: list[str] = []
    for name in (name for name in HF_BASELINES if name in requested):
        needs_drafter = name == "hf_assisted"
        if not _uses_hf_generate(target_module) or (
            needs_drafter and not _uses_hf_generate(drafter_module)
        ):
            print(
                f"Skipping {name}: it needs HuggingFace modules with generate(). "
                "No numbers were written for it."
            )
            continue
        if name == "hf_sample" and greedy:
            print(
                "Skipping hf_sample: temperature is 0, so it would be hf_greedy. "
                "No numbers were written for it."
            )
            continue
        if name == "hf_greedy" and not greedy:
            print(
                f"Skipping hf_greedy: this run samples at temperature {temperature}; "
                "hf_sample is the matching baseline. No numbers were written for it."
            )
            continue
        runnable_hf.append(name)
    methods = [_SPEC] + (["vanilla_ar"] if run_ar else []) + runnable_hf

    target_forwards = ForwardRecorder(target_module)
    drafter_forwards = ForwardRecorder(drafter_module)
    runs = {name: _MethodRuns() for name in methods}
    first_lengths: dict[str, list[int]] = {}

    def _generate(name: str, prompt: torch.Tensor) -> tuple[torch.Tensor, float | None, Any]:
        if name == _SPEC:
            result = engine.generate(prompt, max_new_tokens=max_new_tokens)
            produced = result.output_ids[result.output_ids >= 0].view(1, -1)
            return produced, result.acceptance_rate, result.step_profile
        if name == "vanilla_ar":
            tokens = autoregressive_generate(
                target_module, prompt, max_new_tokens, temperature=temperature
            )
            return tokens, None, None
        tokens = _hf_generate(
            target_module,
            prompt,
            max_new_tokens,
            temperature,
            do_sample=not greedy,
            assistant_model=drafter_module if name == "hf_assisted" else None,
            eos_token_ids=eos_ids if stop_at_eos else None,
        )
        return tokens, None, None

    def _timed(name: str, prompt: torch.Tensor, record: bool) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        target_forwards.clear()
        drafter_forwards.clear()
        target_forwards.attach()
        drafter_forwards.attach()
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            tokens, alpha, steps = _generate(name, prompt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        finally:
            target_forwards.detach()
            drafter_forwards.detach()
        if not record:
            return
        stats = runs[name]
        stats.latency_ms.append(elapsed_ms)
        stats.tokens.append(int(tokens.shape[1]))
        stats.outputs.append([int(token) for token in tokens.view(-1).tolist()])
        if alpha is not None:
            stats.alpha.append(alpha)
        if steps is not None:
            stats.steps.extend(steps)
        stats.target_forwards += target_forwards.calls
        stats.drafter_forwards += drafter_forwards.calls
        if device.type == "cuda":
            stats.peak_memory_bytes = max(
                stats.peak_memory_bytes, int(torch.cuda.max_memory_allocated())
            )
        if name not in first_lengths:
            first_lengths[name] = list(target_forwards.lengths)
            first_lengths[f"{name}_drafter"] = list(drafter_forwards.lengths)

    for prompt in prompts[: max(n_warmup, 0)]:
        for name in methods:
            _timed(name, prompt, record=False)
    for index, prompt in enumerate(prompts):
        # Rotate the order so no method always runs first (or last) on a prompt.
        shift = index % len(methods)
        for name in methods[shift:] + methods[:shift]:
            _timed(name, prompt, record=True)

    # One extra, untimed pass per method on the first prompt: host reads
    # and NaN/inf logits. These probes would disturb the timed runs.
    counter_kind = "cuda_sync_debug_mode" if device.type == "cuda" else "cpu_tensor_read_proxy"
    probes: dict[str, dict[str, object]] = {}
    for name in methods:
        with (
            NonFiniteLogits(target_module) as target_bad,
            NonFiniteLogits(drafter_module) as drafter_bad,
            ForwardRecorder(target_module) as probe_forwards,
        ):
            with count_host_syncs(device) as syncs:
                _tokens, _alpha, probe_steps = _generate(name, prompts[0])
        after_prefill = max(probe_forwards.calls - 1, 0)
        host: dict[str, object] = {
            "counter": counter_kind,
            "prompt_id": prompt_ids[0] if prompt_ids else None,
            "total": syncs.count,
            "target_forwards_after_prefill": after_prefill,
            "per_target_forward": syncs.count / after_prefill if after_prefill else 0.0,
            "by_kind": dict(syncs.by_kind),
        }
        if probe_steps is not None:
            host["n_steps"] = len(probe_steps)
            host["per_step"] = syncs.count / len(probe_steps) if probe_steps else 0.0
        probes[name] = {
            "host_syncs": host,
            "nonfinite_logits": {
                "scope": "the untimed probe generation on the first prompt",
                "prompt_id": prompt_ids[0] if prompt_ids else None,
                "target_entries": target_bad.count,
                "target_forwards_checked": target_bad.calls,
                "drafter_entries": drafter_bad.count,
                "drafter_forwards_checked": drafter_bad.calls,
            },
        }

    spec = runs[_SPEC]

    def _paired(other: _MethodRuns) -> dict[str, float]:
        ratios = [
            (s_n / (s_ms / 1000.0)) / (o_n / (o_ms / 1000.0))
            if s_ms > 0.0 and o_ms > 0.0 and o_n > 0
            else 0.0
            for s_ms, s_n, o_ms, o_n in zip(
                spec.latency_ms, spec.tokens, other.latency_ms, other.tokens, strict=True
            )
        ]
        mean = sum(ratios) / len(ratios)
        var = sum((item - mean) ** 2 for item in ratios) / len(ratios)
        return {"mean": mean, "std": var**0.5, "median": statistics.median(ratios)}

    def _match(candidate: _MethodRuns, reference: _MethodRuns, ref_name: str) -> dict[str, object]:
        firsts: list[int | None] = []
        for got, want in zip(candidate.outputs, reference.outputs, strict=True):
            if got == want:
                firsts.append(None)
                continue
            limit = min(len(got), len(want))
            firsts.append(next((i for i in range(limit) if got[i] != want[i]), limit))
        return {
            "reference": ref_name,
            "prompts_matched": sum(item is None for item in firsts),
            "n_prompts": len(firsts),
            "first_divergence": firsts,
        }

    reference_name = "hf_greedy" if "hf_greedy" in runs else ("vanilla_ar" if run_ar else None)

    def _forward_counts(name: str) -> dict[str, object]:
        stats = runs[name]
        n_runs = len(stats.tokens)
        emitted = sum(stats.tokens)
        after_prefill = stats.target_forwards - n_runs
        counts: dict[str, object] = {
            "n_generations": n_runs,
            "tokens_emitted": emitted,
            "target_forwards": stats.target_forwards,
            "target_forwards_per_emitted_token": (
                stats.target_forwards / emitted if emitted else 0.0
            ),
            "emitted_tokens_per_target_forward_after_prefill": (
                emitted / after_prefill if after_prefill > 0 else 0.0
            ),
            "target_input_lengths_first_generation": first_lengths.get(name, []),
        }
        if name in (_SPEC, "hf_assisted"):
            counts["drafter_forwards"] = stats.drafter_forwards
            counts["drafter_input_lengths_first_generation"] = first_lengths.get(
                f"{name}_drafter", []
            )
        if name == _SPEC:
            n_steps = len(stats.steps)
            counts["n_steps"] = n_steps
            counts["target_forwards_per_step"] = (
                (stats.target_forwards - n_runs) / n_steps if n_steps else 0.0
            )
            counts["drafter_forwards_per_step"] = (
                (stats.drafter_forwards - n_runs) / n_steps if n_steps else 0.0
            )
        return counts

    def _metrics(name: str) -> dict[str, float]:
        stats = runs[name]
        metrics = summarise_runs(
            stats.latency_ms, stats.tokens, stats.alpha if name == _SPEC else None
        )
        metrics.pop("gpu_memory_gb", None)
        if device.type == "cuda":
            metrics["gpu_memory_gb"] = stats.peak_memory_bytes / (1024**3)
        return metrics

    hardware = detect_hardware()
    baseline_unmeasured = [
        "acceptance_rate_mean",
        "acceptance_rate_std",
        "speedup_vs_ar_mean",
        "speedup_vs_ar_std",
    ]
    if device.type != "cuda":
        baseline_unmeasured.append("gpu_memory_gb")
    unregister(drafter_name)

    spec_metrics = _metrics(_SPEC)
    if run_ar:
        paired = _paired(runs["vanilla_ar"])
        spec_metrics["speedup_vs_ar_mean"] = paired["mean"]
        spec_metrics["speedup_vs_ar_std"] = paired["std"]
    for name in runnable_hf:
        paired = _paired(runs[name])
        spec_metrics[f"speedup_vs_{name}_mean"] = paired["mean"]
        spec_metrics[f"speedup_vs_{name}_std"] = paired["std"]
        spec_metrics[f"speedup_vs_{name}_median"] = paired["median"]
    spec_unmeasured = [] if device.type == "cuda" else ["gpu_memory_gb"]
    if not run_ar:
        spec_unmeasured += ["speedup_vs_ar_mean", "speedup_vs_ar_std"]
    spec_extra: dict[str, object] = {
        **common_extra,
        "step_profile": summarise_steps(spec.steps),
        "forward_counts": _forward_counts(_SPEC),
        **probes[_SPEC],
    }
    if greedy and reference_name is not None:
        spec_extra["greedy_token_match"] = _match(spec, runs[reference_name], reference_name)

    documents: dict[str, dict[str, object]] = {
        _SPEC: assemble_result(
            model=model_name,
            method=_SPEC,
            gamma=gamma,
            dataset=dataset,
            n_samples=len(prompts),
            metrics=spec_metrics,
            measurement_kind=kind,
            notes=spec_notes,
            unmeasured=spec_unmeasured,
            hardware=hardware,
            prompt_ids=prompt_ids,
            extra=spec_extra,
        )
    }
    mode = "greedy" if greedy else f"sampled (temperature {temperature})"
    method_notes = {
        "vanilla_ar": ar_notes,
        "hf_greedy": "HuggingFace generate(do_sample=False) with the KV cache. ",
        "hf_sample": f"HuggingFace generate(do_sample=True, temperature={temperature}). ",
        "hf_assisted": (
            f"HuggingFace {mode} generate(assistant_model=drafter) with the "
            "assistant's default schedule: " + _assistant_schedule(drafter_module) + ". "
        ),
    }
    for name in methods[1:]:
        extra: dict[str, object] = {
            **common_extra,
            "forward_counts": _forward_counts(name),
            **probes[name],
        }
        if greedy and reference_name is not None and name not in (reference_name, "vanilla_ar"):
            extra["greedy_token_match"] = _match(runs[name], runs[reference_name], reference_name)
        notes = method_notes[name]
        if name != "vanilla_ar":
            notes += "Same target, prompts and max_new_tokens. " + shared_notes
        documents[name] = assemble_result(
            model=model_name,
            method=name,
            gamma=gamma,
            dataset=dataset,
            n_samples=len(prompts),
            metrics=_metrics(name),
            measurement_kind=kind,
            notes=notes,
            unmeasured=list(baseline_unmeasured),
            hardware=hardware,
            prompt_ids=prompt_ids,
            extra=extra,
        )
    return documents


def _assistant_schedule(drafter: nn.Module) -> str:
    """Describe the assistant settings HuggingFace assisted generation will use."""
    generation_config = getattr(drafter, "generation_config", None)
    keys = (
        "num_assistant_tokens",
        "num_assistant_tokens_schedule",
        "assistant_confidence_threshold",
    )
    # Transformers 5 leaves these unset on a fresh GenerationConfig and
    # resolves them inside generate(); report the value it will use.
    defaults: dict[str, object] = {}
    resolver = getattr(type(generation_config), "_get_default_generation_params", None)
    if callable(resolver):
        try:
            defaults = dict(resolver())
        except Exception:  # noqa: BLE001 - a private helper; fall back to unset
            defaults = {}
    parts: list[str] = []
    for key in keys:
        value = getattr(generation_config, key, None)
        if value is None and key in defaults:
            parts.append(f"{key}={defaults[key]} (library default)")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _dataset_names(config: dict[str, Any]) -> list[str]:
    """Return canonical dataset names from ``datasets`` or ``dataset``.

    Parameters
    ----------
    config : dict
        Benchmark config.

    Returns
    -------
    list[str]
        One slug per requested dataset, in file order.
    """
    raw = config.get("datasets")
    if isinstance(raw, list) and raw:
        return [canonical_dataset(str(item)) for item in raw]
    return [canonical_dataset(str(config.get("dataset", "synthetic_prompts")))]


def _temperatures(config: dict[str, Any]) -> list[float] | None:
    """Return the ``temperatures`` list, or ``None`` for a single-mode config."""
    raw = config.get("temperatures")
    if isinstance(raw, list) and raw:
        return [float(item) for item in raw]
    return None


def _mode_slug(temperature: float) -> str:
    """``greedy`` for temperature 0, otherwise ``t<temperature>`` (``t0.7``)."""
    return "greedy" if temperature == 0.0 else f"t{temperature:g}"


def _result_filename(
    stem: str,
    dataset: str,
    method: str,
    commit: str,
    *,
    multiple: bool = False,
    mode: str | None = None,
) -> str:
    """Return the JSON file name for one method and dataset.

    Parameters
    ----------
    stem : str
        Config file stem.
    dataset : str
        Canonical dataset slug.
    method : str
        Method name.
    commit : str
        Git commit recorded in the file.
    multiple : bool
        True when the config runs more than one dataset.
    mode : str or None
        Decoding mode slug from a ``temperatures`` list (``greedy``,
        ``t0.7``). It goes after the dataset slug.

    Returns
    -------
    str
        ``{stem}_{method}_{commit}.json`` for one synthetic dataset.
        The dataset slug is included for a real prompt set or when more
        than one dataset is run, so the files do not overwrite each other.
    """
    slug = canonical_dataset(dataset)
    if mode is not None:
        return f"{stem}_{slug}_{mode}_{method}_{commit}.json"
    if multiple or slug in REAL_DATASETS:
        return f"{stem}_{slug}_{method}_{commit}.json"
    return f"{stem}_{method}_{commit}.json"


def run_config_file(path: Path, output_dir: Path) -> list[Path]:
    """Run one YAML config and write measured JSON files.

    Medusa and EAGLE entries in ``methods`` are skipped with no output
    file. A CUDA config on a machine without CUDA raises
    :class:`DeviceUnavailableError` before any file is written for that
    dataset. A ``datasets`` list is run in order. Alpaca is skipped
    when the loop reaches it. A list that contains only Alpaca raises
    and writes nothing.

    Parameters
    ----------
    path : Path
        Config YAML.
    output_dir : Path
        Directory for JSON results.

    Returns
    -------
    list[Path]
        Files written.

    Raises
    ------
    RuntimeError
        If no dataset remains, or a real prompt set cannot be tokenized.
        Nothing is written for the dataset that failed.
    """
    config = load_benchmark_config(path)
    methods = config.get("methods") or ["flashspec_ucb", "vanilla_ar"]
    skipped = [name for name in methods if name in {"medusa", "eagle", "flashspec_thompson"}]
    if skipped:
        print(
            "Skipping methods with no implementation in this repository: "
            + ", ".join(skipped)
            + ". No numbers were written for them."
        )
    if "flashspec_ucb" not in methods and "vanilla_ar" not in methods:
        raise RuntimeError(f"{path.name} does not request a method this harness can measure.")
    names = _dataset_names(config)
    planned = [name for name in names if name != "alpaca"]
    if not planned:
        for name in names:
            if name == "alpaca":
                print(f"Skipping alpaca: {ALPACA_REASON}")
        raise RuntimeError(ALPACA_REASON)
    multiple = len(planned) > 1
    written: list[Path] = []
    commit = git_commit()
    temperatures = _temperatures(config)
    modes: list[float | None] = list(temperatures) if temperatures else [None]
    # Checkpoints are loaded once per config, not once per dataset and mode.
    modules: dict[str, Any] = {}
    if str(config.get("model", "toy")) not in TOY_MODELS:
        device = resolve_device(str(config.get("device", "auto")))
        drafter_module, target_module = _load_checkpoint_modules(config, device)
        modules = {"drafter_module": drafter_module, "target_module": target_module}
        if any(item in REAL_DATASETS for item in planned) and config.get("tokenizer") is None:
            config = {**config, "tokenizer": _load_tokenizer(str(config["model"]))}
    for name in names:
        if name == "alpaca":
            print(f"Skipping alpaca: {ALPACA_REASON}")
            continue
        for temperature in modes:
            one = {
                key: value
                for key, value in config.items()
                if key not in {"datasets", "temperatures"}
            }
            one["dataset"] = name
            if temperature is not None:
                one["temperature"] = temperature
            mode = _mode_slug(temperature) if temperature is not None else None
            print(f"Running {path.stem}: dataset={name} mode={mode or one.get('temperature')}")
            paired = run_paired_generations(one, **modules)
            for method, document in paired.items():
                destination = output_dir / _result_filename(
                    path.stem, name, method, commit, multiple=multiple, mode=mode
                )
                write_result(destination, document)
                written.append(destination)
                metrics = document["metrics"]
                assert isinstance(metrics, dict)
                print(
                    f"{method}: {metrics.get('tokens_per_second_median')} tok/s (median) "
                    f"on {document['hardware']}"
                )
    return written
