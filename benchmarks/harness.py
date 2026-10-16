"""Reproducible benchmark helpers.

Every result dict written by this module sets ``status`` to ``"measured"``
or is not written at all. Fields that were not measured are omitted and
listed under ``unmeasured``. Hardware is detected at runtime. A config
that asks for CUDA when CUDA is absent raises, and no JSON is written.

Medusa and EAGLE are not implemented in this repository. Callers must
not invent numbers for them.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
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
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig
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
    metrics: dict[str, float] = {
        "tokens_per_second_mean": mean_tps,
        "tokens_per_second_std": var_tps**0.5,
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
) -> torch.Tensor:
    """Sample with ``model.generate`` and the model's own KV cache.

    Parameters
    ----------
    model : nn.Module
        HuggingFace causal LM.
    input_ids : torch.Tensor
        Prompt. Shape ``(batch, seq)``.
    max_new_tokens : int
        Number of new tokens. The call does not stop early on EOS.
    temperature : float
        Sampling temperature passed through to ``generate``.

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
        "do_sample": True,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": max_new_tokens,
        "use_cache": True,
        "eos_token_id": None,
    }
    if pad is not None:
        kwargs["pad_token_id"] = pad
    with torch.no_grad():
        output = model.generate(input_ids, **kwargs)
    new_tokens = output[:, prompt_len:]
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
        scaled = logits if temperature == 1.0 else logits / temperature
        next_id = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
        pieces.append(next_id)
        logits = session.append(next_id)[:, -1, :]
    return torch.cat(pieces, dim=1)


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
        Applied to logits before softmax.

    Returns
    -------
    torch.Tensor
        New token ids. Shape ``(batch, max_new_tokens)``.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}.")
    if _uses_hf_generate(model):
        return _hf_generate(model, input_ids, max_new_tokens, temperature)
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
        if temperature != 1.0:
            logits = logits / temperature
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
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
    prompts = tokenize_prompts(chosen, tokenizer, device, limit)
    ids = [item["id"] for item in chosen]
    note = (
        f" Prompts are the fixed {slug} set ({', '.join(ids)}), tokenized "
        "with the target tokenizer."
    )
    if capped:
        note += (
            f" {source} asked for {requested} prompts; the fixed set has "
            f"{len(records)}, so each prompt is used once."
        )
    if str(config.get("model", "toy")) == "toy":
        note += (
            " The network is randomly initialised, so acceptance is not a checkpoint measurement."
        )
    return prompts, ids, note


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
    toy = model_name == "toy"
    if dataset == "alpaca":
        raise RuntimeError(ALPACA_REASON)
    if dataset in REAL_DATASETS and tokenizer is None and toy:
        raise RuntimeError("A real prompt set requires a tokenizer. No result file was written.")
    seed = int(config.get("seed", 0))
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if drafter_module is None or target_module is None:
        if toy:
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
    return drafter.to(device), target.to(device)


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
    drafters = load_drafters(
        [drafter_name],
        device=device,
        dtype=weight_dtype,
        temperature=temperature,
        models={drafter_name: drafter_module},
    )
    target = TargetModel(model_name, device, weight_dtype, model=target_module)
    engine_config = FlashSpecConfig(
        drafter_name=drafter_name,
        target_name=model_name,
        device=str(device),
        dtype=str(weight_dtype).removeprefix("torch."),
        max_new_tokens=max_new_tokens,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=gamma, temperature=temperature, seed=seed),
    )
    engine = SpeculativeEngine(
        engine_config,
        drafters,
        target,
        UCB1Selector(n_arms=1, window_size=0),
    )
    prompts, prompt_ids, prompt_note = _prompts(config, device, tokenizer)
    kind = "toy_random_init_transformer" if toy else "end_to_end"
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

    def _time_speculative(prompt: torch.Tensor) -> tuple[float, int, float]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = engine.generate(prompt, max_new_tokens=max_new_tokens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, result.n_tokens_generated, result.acceptance_rate

    def _time_ar(prompt: torch.Tensor) -> tuple[float, int]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        tokens = autoregressive_generate(
            target_module, prompt, max_new_tokens, temperature=temperature
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, int(tokens.shape[1])

    for prompt in prompts[: max(n_warmup, 0)]:
        _time_speculative(prompt)
        _time_ar(prompt)

    spec_latency: list[float] = []
    spec_tokens: list[int] = []
    spec_alpha: list[float] = []
    ar_latency: list[float] = []
    ar_tokens: list[int] = []
    for prompt in prompts:
        latency_ms, n_tokens, alpha = _time_speculative(prompt)
        spec_latency.append(latency_ms)
        spec_tokens.append(n_tokens)
        spec_alpha.append(alpha)
        ar_ms, ar_n = _time_ar(prompt)
        ar_latency.append(ar_ms)
        ar_tokens.append(ar_n)

    spec_metrics = summarise_runs(spec_latency, spec_tokens, spec_alpha)
    ar_metrics = summarise_runs(ar_latency, ar_tokens, None)
    ratios = [
        (spec_n / (spec_ms / 1000.0)) / (ar_n / (ar_ms / 1000.0))
        if spec_ms > 0.0 and ar_ms > 0.0 and ar_n > 0
        else 0.0
        for spec_ms, spec_n, ar_ms, ar_n in zip(
            spec_latency, spec_tokens, ar_latency, ar_tokens, strict=True
        )
    ]
    mean_ratio = sum(ratios) / len(ratios)
    var_ratio = sum((item - mean_ratio) ** 2 for item in ratios) / len(ratios)
    spec_metrics["speedup_vs_ar_mean"] = mean_ratio
    spec_metrics["speedup_vs_ar_std"] = var_ratio**0.5

    hardware = detect_hardware()
    spec_unmeasured = [name for name in ("gpu_memory_gb",) if name not in spec_metrics]
    ar_unmeasured = [
        "acceptance_rate_mean",
        "acceptance_rate_std",
        "speedup_vs_ar_mean",
        "speedup_vs_ar_std",
    ]
    if "gpu_memory_gb" not in ar_metrics:
        ar_unmeasured.append("gpu_memory_gb")
    unregister(drafter_name)
    return {
        "flashspec_ucb": assemble_result(
            model=model_name,
            method="flashspec_ucb",
            gamma=gamma,
            dataset=dataset,
            n_samples=len(prompts),
            metrics=spec_metrics,
            measurement_kind=kind,
            notes=spec_notes,
            unmeasured=spec_unmeasured,
            hardware=hardware,
            prompt_ids=prompt_ids,
        ),
        "vanilla_ar": assemble_result(
            model=model_name,
            method="vanilla_ar",
            gamma=gamma,
            dataset=dataset,
            n_samples=len(prompts),
            metrics=ar_metrics,
            measurement_kind=kind,
            notes=ar_notes,
            unmeasured=ar_unmeasured,
            hardware=hardware,
            prompt_ids=prompt_ids,
        ),
    }


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


def _result_filename(
    stem: str,
    dataset: str,
    method: str,
    commit: str,
    *,
    multiple: bool = False,
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

    Returns
    -------
    str
        ``{stem}_{method}_{commit}.json`` for one synthetic dataset.
        The dataset slug is included for a real prompt set or when more
        than one dataset is run, so the files do not overwrite each other.
    """
    slug = canonical_dataset(dataset)
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
    for name in names:
        if name == "alpaca":
            print(f"Skipping alpaca: {ALPACA_REASON}")
            continue
        one = {key: value for key, value in config.items() if key != "datasets"}
        one["dataset"] = name
        paired = run_paired_generations(one)
        for method, document in paired.items():
            destination = output_dir / _result_filename(
                path.stem, name, method, commit, multiple=multiple
            )
            write_result(destination, document)
            written.append(destination)
            metrics = document["metrics"]
            assert isinstance(metrics, dict)
            print(
                f"{method}: {metrics.get('tokens_per_second_mean')} tok/s on {document['hardware']}"
            )
    return written
