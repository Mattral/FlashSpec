# FlashSpec

Lossless speculative decoding for HuggingFace causal language models, with a static-cache CUDA-graph fast path.

[![PyPI](https://img.shields.io/pypi/v/flashspec)](https://pypi.org/project/flashspec/)
[![Python](https://img.shields.io/pypi/pyversions/flashspec)](https://pypi.org/project/flashspec/)
[![CI](https://github.com/Mattral/FlashSpec/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/FlashSpec/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://flashspec.readthedocs.io)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20766119-blue)](https://doi.org/10.5281/zenodo.20766119)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Mattral/FlashSpec/blob/main/LICENSE)

## What it is

FlashSpec wraps an existing HuggingFace causal LM and a smaller drafter that shares its tokenizer: the drafter proposes tokens, your model checks them in one forward pass, and the output is your model's own: the same distribution when sampling, and the same tokens in greedy mode up to fp16 ties ([correctness](https://github.com/Mattral/FlashSpec#correctness)). On one Tesla T4, with `Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct` at batch 1, with the default settings, it was faster than HuggingFace `generate` compiled the same way in every measured cell, by at least 1.5x on HumanEval and by less on MT-Bench ([results](https://github.com/Mattral/FlashSpec#results)). That is the whole measured scope (one model pair, one GPU, batch 1), and FlashSpec is a single-device library with a small HTTP server, not a serving stack.

## When to use it

Use it when:

- you run a HuggingFace causal LM on a CUDA GPU, one request at a time (batch 1);
- you have a smaller drafter with the same tokenizer, or your outputs copy from the prompt (code edits, quoting), where the built-in prompt-lookup drafter needs no second model;
- you need your model's exact output, not an approximation of it.

Not yet:

- **CPU.** Both recorded CPU end-to-end runs were slower than plain decoding ([history](https://github.com/Mattral/FlashSpec/blob/main/docs/benchmarks.md#historical-and-microbenchmark-results)).
- **Batch > 1.** Larger batches use a generic loop that reads the device several times per step and has not been measured.
- **First-call latency.** On CUDA the first `generate` compiles and captures CUDA graphs, which can take minutes. `compile=False` skips that, but in the 0.3.0 run the eager path, while faster than plain HuggingFace `generate` in all 4 cells, was slower than compiled HuggingFace in all 4.
- **Other GPUs and model pairs.** They have not been measured.

## Install

```bash
pip install flashspec
```

| extra | adds | when you want it |
|---|---|---|
| `flashspec[gpu]` | Triton (Linux only) | The compiled path needs Triton; Linux CUDA builds of PyTorch already ship it. Also enables the experimental Triton kernels. |
| `flashspec[serve]` | FastAPI, uvicorn, httpx, accelerate | The OpenAI-compatible server (it loads checkpoints with `device_map`, which needs `accelerate`). |
| `flashspec[asm]` | `flashspec-asm-kernel` (Linux x86-64) | An experimental CPU residual kernel. It does not change end-to-end speed. |

You need Python 3.11 or later, PyTorch 2.2 or later and transformers 4.40 or later. Installing from source, Windows and macOS notes, and the full requirements table are in [docs/install.md](https://github.com/Mattral/FlashSpec/blob/main/docs/install.md).

**Container image (GPU-oriented).** Each release is also published as a server image on GitHub Container Registry. It is built from the PyPI release on CUDA 13.0, with the torch build the GPU results were measured with, and it runs as a non-root user. It needs an NVIDIA GPU and the NVIDIA Container Toolkit; it starts on CPU too, but CPU is not where FlashSpec is faster. The default command serves the measured pair in fp16 on port 8000.

```bash
docker pull ghcr.io/mattral/flashspec:0.3.2
docker run --gpus all -p 8000:8000 ghcr.io/mattral/flashspec:0.3.2
```

## Quickstart

This loads the measured model pair, decodes greedily, and prints the text and the speed. With one model drafter the default `drafter_strategy="auto"` runs `composite-g6-f2`: prompt lookup when the context has a match, else the drafter (see [Choose a drafter](https://github.com/Mattral/FlashSpec#choose-a-drafter)). Without CUDA it falls back to CPU and float32, which is slow.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashspec import FlashSpecConfig, SamplingConfig, SpeculativeEngine
from flashspec.engine import HuggingFaceDrafter, TargetModel
target_id, drafter_id = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"  # same tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
load = lambda name: AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
target = TargetModel(target_id, device, dtype, model=load(target_id))
drafter = HuggingFaceDrafter(drafter_id, device, dtype, model=load(drafter_id))
config = FlashSpecConfig(device=str(device), sampling=SamplingConfig(temperature=0.0))  # greedy
engine = SpeculativeEngine(config, [drafter], target)
tokenizer = AutoTokenizer.from_pretrained(target_id)
prompt = tokenizer("def fibonacci(n):", return_tensors="pt").input_ids.to(device)
engine.generate(prompt, max_new_tokens=128)  # the first call on CUDA compiles (can take minutes)
result = engine.generate(prompt, max_new_tokens=128)
print(tokenizer.decode(result.output_ids[0, : result.n_tokens_generated], skip_special_tokens=True))
print(f"{result.tokens_per_second:.1f} tok/s, acceptance {result.acceptance_rate:.2f}")
print("drafter strategy:", engine.drafter_strategy)  # composite-g6-f2 with one model drafter
```

`output_ids` has shape `(batch, max_new_tokens)` and pads a row that stopped early with `-1`; `n_tokens_generated` counts the real tokens. `TargetModel` and `HuggingFaceDrafter` take the loaded modules through `model=`; without it, `TargetModel` loads with `device_map`, which needs `accelerate`.

## Recipes

The recipes reuse `target`, `drafter`, `device`, `prompt` and `tokenizer` from the quickstart.

### Choose a drafter

There are three kinds, and all are lossless:

- a **model drafter** (`HuggingFaceDrafter`), as in the quickstart;
- **prompt lookup** (`NgramDrafter`): proposes the continuation of the most recent earlier match of the last 1 to 3 tokens. No second model;
- **composite**: prompt lookup when the context has a match, else the model drafter.

`FlashSpecConfig.drafter_strategy` names the arms the engine builds when you do not pass `arms=` (the server takes the same names as `--drafter-strategy`):

| strategy | what drafts each step | drafters to pass |
|---|---|---|
| `"auto"` (default) | `"composite-g6-f2"` when you pass one model drafter, the `"bandit"` path otherwise; a call at batch > 1 or with `variant="typical"` uses the model drafter at `sampling.gamma` | one or more drafters |
| `"bandit"` (0.2.0 behaviour) | one arm per drafter at `sampling.gamma`, chosen online by the selector you pass (experimental) | one or more drafters |
| `"composite-g4"` | prompt lookup, up to 4 tokens, when the context has a match; else the model drafter drafts 4 | a model drafter (an `NgramDrafter` is added if you pass none) |
| `"composite-g6-f2"` | prompt lookup, up to 6 tokens, when the context has a match; else the model drafter drafts 2 | a model drafter (an `NgramDrafter` is added if you pass none) |

A composite strategy is one fixed arm (`bandit.n_arms` defaults to 1), runs at batch 1 with `variant` `"rejection"` or `"greedy"` (set explicitly, any other call raises), and verifies at width `max gamma + 1`. Lookup steps use Algorithm 1 with a one-hot draft distribution, so it is lossless; the chi-squared tests cover separate lookup and fallback gammas. The engine logs one line with the strategy it chose and why, and keeps them in `engine.drafter_strategy` and `engine.drafter_strategy_reason`. `drafter_strategy="bandit"` restores the 0.2.0 behaviour.

The default was chosen by a rule fixed before the run (`benchmarks/configs/qwen_t4_composite.yaml`, applied by `scripts/composite_report.py`). In that T4 run (bundle `fdd3be9`), composite-g6-f2 had the higher mixed-workload greedy throughput, was faster than compiled HuggingFace on HumanEval, MT-Bench and HumanEval-edit in greedy and at T=0.7, and matched compiled HuggingFace greedy output except at three fp16 exact ties (HumanEval/53, MT-Bench 81 and 101, the ties the #47 diagnostic found) ([values](https://github.com/Mattral/FlashSpec/tree/main/benchmarks/results/gpu/fdd3be9)). The README tables below are the 0.3.0 release run with this default (bundle `3c43e08`).

```python
from flashspec import BanditConfig
from flashspec.engine import NgramDrafter

greedy = SamplingConfig(temperature=0.0)
composite = FlashSpecConfig(device=str(device), drafter_strategy="composite-g6-f2", sampling=greedy)
engine = SpeculativeEngine(composite, [drafter, NgramDrafter()], target)
result = engine.generate(prompt, max_new_tokens=64)
print(engine.arms, [step.drafter for step in result.arm_steps])  # drafter used at each step

lookup_only = FlashSpecConfig(device=str(device), drafter_strategy="bandit",  # one arm per drafter
                              bandit=BanditConfig(n_arms=1), sampling=greedy)
lookup = SpeculativeEngine(lookup_only, [NgramDrafter()], target)
```

Which is fastest depends on the prompts. In the Phase 6 T4 run (greedy, bundle `3e38251`), the fastest fixed choice was the composite at gamma 4 on HumanEval and on the mixed workload, the model drafter at gamma 2 on MT-Bench, and prompt lookup at gamma 6 on code edits ([values](https://github.com/Mattral/FlashSpec/tree/main/benchmarks/results/gpu/3e38251)). Several arms plus a selector (`UCB1Selector(n_arms=k)` or `ThompsonSelector(n_arms=k)` from `flashspec.bandit`, passed as `SpeculativeEngine`'s fourth argument) let a bandit choose online; with one arm the engine needs no selector; that is experimental, and in the same run it did not beat the best fixed arm ([bandit](https://github.com/Mattral/FlashSpec/blob/main/docs/bandit.md)).

### Greedy or sampling

```python
greedy = SamplingConfig(temperature=0.0)            # same tokens as model.generate(do_sample=False)
sampled = SamplingConfig(temperature=0.7, seed=0)   # Algorithm 1: same distribution as the target
drafter.set_temperature(0.7)                        # draft at the target's temperature
config = FlashSpecConfig(device=str(device), sampling=sampled)
engine = SpeculativeEngine(config, [drafter], target)
result = engine.generate(prompt, max_new_tokens=64)
```

The default is `temperature=1.0`, which samples. `temperature=0.0` selects greedy. `seed` makes one call reproducible; `seed=None` uses the global RNG. `variant="typical"` is lossy and is not on the fast path.

### Turn compile off

```python
config = FlashSpecConfig(device=str(device), compile=False, sampling=SamplingConfig(temperature=0.0))
engine = SpeculativeEngine(config, [drafter], target)
print(engine.compiled, "|", engine.compile_decision.reason)
```

The default `compile="auto"` turns the compiled path on when the device is CUDA and Triton is installed. The engine logs one line saying which path runs and why. Set `compile=False` before a compiled engine has run on the same `TargetModel` and drafter: the compiled path moves those wrappers onto static caches, and they stay there. The server takes `--compile off`.

### Run the OpenAI-compatible server

```bash
pip install "flashspec[serve]"
python -m flashspec.serve --target Qwen/Qwen2.5-1.5B-Instruct --drafter Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 --dtype float16 --temperature 0
```

Or with the container image. Arguments replace its default command, so pass `--host 0.0.0.0`. The volume keeps downloaded checkpoints between runs:

```bash
docker run --gpus all -p 8000:8000 -v flashspec-hf:/home/flashspec/.cache/huggingface ghcr.io/mattral/flashspec:0.3.2 \
    --target Qwen/Qwen2.5-1.5B-Instruct --drafter Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 --dtype float16 --temperature 0 --host 0.0.0.0
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Name three primary colors."}], "max_tokens": 32}'
```

The server has `POST /v1/chat/completions`, `POST /v1/completions`, `GET /health` and `GET /metrics`. It runs one process on one device, with no streaming and no continuous batching. Temperature and gamma are fixed at start, and a request with another temperature is rejected. `GET /health` lists every limitation.

### Use your own drafter

Any HuggingFace causal LM with the target's tokenizer works as `HuggingFaceDrafter(model_id, device, dtype, model=...)`. Anything else implements the `DraftModel` protocol: `generate_draft` returns `gamma` proposed ids and the log-probability rows they were drawn from, shape `(batch, gamma, vocab)` over the target's vocabulary, float32.

```python
class RepeatLast:
    """Proposes the last token gamma times; its q is one-hot, so Algorithm 1 stays exact."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def generate_draft(self, input_ids: torch.Tensor, gamma: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = input_ids[:, -1:].repeat(1, gamma)
        return ids, self.compute_logprobs(input_ids, ids)

    def compute_logprobs(self, input_ids: torch.Tensor, draft_token_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(draft_token_ids, self.vocab_size).float().log()

own = SpeculativeEngine(config, [RepeatLast(target.vocab_size)], target)
result = own.generate(prompt, max_new_tokens=32)
```

With the default `drafter_strategy="auto"`, one drafter of yours becomes the fallback of the composite (prompt lookup first); pass `drafter_strategy="bandit"` to run it alone. The proposals must be draws from the distribution you return. Algorithm 1 is then exact for any drafter, and a poor drafter only lowers the acceptance rate. To build a drafter by name with `load_drafters`, decorate its class with `@flashspec.register("name")`; the constructor then receives `device`, `dtype` and `temperature`.

## API at a glance

The compute dtype is the one the model wrappers load with or are given; `FlashSpecConfig.dtype` is recorded with the config and not applied.

<!-- BEGIN GENERATED: api (scripts/readme_api.py) -->
Generated from the code by `scripts/readme_api.py`; a test fails when it is stale.

| API | What it does | Key arguments, with defaults |
|---|---|---|
| `flashspec.FlashSpecConfig` | Top-level FlashSpec runtime configuration. | fields: `drafter_name: str = "llama3-1b"`, `target_name: str = "meta-llama/Llama-3-8B-Instruct"`, `device: str = "cuda:0"`, `dtype: "float32" \| "bfloat16" \| "float16" = "bfloat16"`, `max_new_tokens: int = 512`, `eos_token_ids: tuple[int, ...] = ()`, `drafter_strategy: "auto" \| "bandit" \| "composite-g4" \| "composite-g6-f2" = "auto"`, `compile: bool \| "auto" = "auto"`, `compile_max_cache_len: int = 1024`, `compile_fixed_shapes: bool = True`, `bandit: BanditConfig = BanditConfig()`, `sampling: SamplingConfig = SamplingConfig()`, `metrics: MetricsConfig = MetricsConfig()` |
| `flashspec.SamplingConfig` | Configuration for speculative sampling. | fields: `gamma: int = 4`, `temperature: float = 1.0`, `top_p: float = 1.0`, `seed: int \| None = None`, `variant: "rejection" \| "typical" \| "greedy" = "rejection"` |
| `flashspec.BanditConfig` | Configuration for the online bandit draft selector. | fields: `strategy: "ucb1" \| "thompson" \| "oracle" = "ucb1"`, `window_size: int = 500`, `n_arms: int = 2`, `ucb_exploration_constant: float = 1.0`, `thompson_prior_alpha: float = 1.0`, `thompson_prior_beta: float = 1.0` |
| `flashspec.SpeculativeEngine` | Adaptive speculative-decoding inference engine. | arguments: `config: FlashSpecConfig`, `drafters: list[DraftModel]`, `target: TargetModel`, `bandit: DraftSelector \| None = None`, `arms: Sequence[ArmSpec] \| None = None` |
| `flashspec.SpeculativeEngine.generate` | Run the speculative decoding generation loop. | arguments: `input_ids: torch.Tensor`, `max_new_tokens: int \| None = None`, `eos_token_ids: Sequence[int] \| None = None` |
| `flashspec.GenerationResult` | Result of a single `SpeculativeEngine.generate` call. | fields: `output_ids: torch.Tensor`, `n_tokens_generated: int`, `acceptance_rate: float`, `tokens_per_second: float`, `step_profile: tuple[StepRecord, ...] \| None = None`, `arm_steps: tuple[ArmStep, ...] = ()` |
| `flashspec.engine.TargetModel` | Wrapper around a HuggingFace CausalLM used as the speculative verifier. | arguments: `model_name_or_path: str`, `device: torch.device`, `dtype: torch.dtype = torch.bfloat16`, `model: Any \| None = None` |
| `flashspec.engine.HuggingFaceDrafter` | Draft model backed by a HuggingFace causal LM. | arguments: `model_name_or_path: str`, `device: torch.device \| str`, `dtype: torch.dtype = torch.float32`, `temperature: float = 1.0`, `model: Any \| None = None` |
| `flashspec.engine.load_drafters` | Instantiate drafters in arm order. | arguments: `names: list[str] \| tuple[str, ...]`, `device: torch.device \| str`, `dtype: torch.dtype = torch.float32`, `temperature: float = 1.0`, `models: dict[str, Any] \| None = None` |
| `flashspec.engine.NgramDrafter` | Prompt-lookup drafter: continue the most recent earlier match of the last n tokens. | arguments: `max_ngram: int = 3`, `min_ngram: int = 1`, `name: str = "ngram"` |
| `flashspec.engine.ArmSpec` | One bandit arm. | fields: `drafter: int`, `gamma: int`, `label: str = ""`, `fallback: int \| None = None`, `fallback_gamma: int \| None = None` |
| `flashspec.bandit.UCB1Selector` | UCB1 bandit selector for adaptive draft-model selection. | arguments: `n_arms: int`, `window_size: int = 500`, `exploration_constant: float = 1.0`, `normalise_rewards: bool = True` |
| `flashspec.bandit.ThompsonSelector` | Thompson sampling bandit selector for draft-model selection. | arguments: `n_arms: int`, `window_size: int = 500`, `prior_alpha: float = 1.0`, `prior_beta: float = 1.0`, `seed: int \| None = None` |
| `flashspec.engine.DraftModel` | Protocol that every draft model must satisfy. | methods: `generate_draft(input_ids: torch.Tensor, gamma: int)`, `compute_logprobs(input_ids: torch.Tensor, draft_token_ids: torch.Tensor)` |
| `flashspec.register` | Class decorator that registers a draft model under `name`. | arguments: `name: str`, `replace: bool = False` |
| `flashspec.TRITON_AVAILABLE` | `True` only on Linux with Triton installed | constant: `bool` |

Defaults that change behaviour:

- `FlashSpecConfig.compile = "auto"`: the compiled path (static KV cache, CUDA graphs) turns on when the device is CUDA and Triton is installed, and stays off otherwise.
- `SamplingConfig.temperature = 1.0` with `variant = "rejection"`: the default samples (Algorithm 1). `temperature=0.0` selects `variant="greedy"`.
- `SamplingConfig.gamma = 4`: draft tokens per step.
- `FlashSpecConfig.drafter_strategy = "auto"`: the arms `SpeculativeEngine(..., arms=None)` builds. `"auto"` runs `"composite-g6-f2"` (prompt lookup when the context has a match, else the model drafter; an `NgramDrafter` is added) when given one model drafter, and the `"bandit"` path otherwise; a call at batch > 1 or with `variant="typical"` uses the model drafter at `sampling.gamma`. `"bandit"` (the 0.2.0 behaviour) is one arm per drafter at `sampling.gamma`, chosen by the selector you pass. An explicit composite needs batch 1.
- `BanditConfig.n_arms = 2` when you pass a `BanditConfig` yourself: it must equal the number of arms, so pass `BanditConfig(n_arms=1)` for one arm. Under `"auto"` with no `bandit=`, or a composite with no `bandit=`, it is not a constraint.
- `FlashSpecConfig.device = "cuda:0"`, `max_new_tokens = 512`.

Server flags (`python -m flashspec.serve`):

| flag | default | choices |
|---|---|---|
| `--target` | required |  |
| `--drafter` (repeatable) | required |  |
| `--device` | `"cpu"` |  |
| `--host` | `"127.0.0.1"` |  |
| `--port` | `8000` |  |
| `--batch-size` | `1` |  |
| `--max-wait-ms` | `2.0` |  |
| `--max-queue` | `32` |  |
| `--gamma` | `4` |  |
| `--max-new-tokens` | `128` |  |
| `--temperature` | `1.0` |  |
| `--drafter-strategy` | `"auto"` | "auto", "bandit", "composite-g4", "composite-g6-f2" |
| `--strategy` | `"ucb1"` | "ucb1", "thompson" |
| `--dtype` | `"float32"` | "float32", "bfloat16", "float16" |
| `--compile` | `"auto"` | "auto", "on", "off" |
<!-- END GENERATED: api -->

The full reference is generated from the docstrings: [API reference](https://flashspec.readthedocs.io/en/latest/api/).

## Results

<!-- BEGIN GENERATED: status (scripts/ingest_gpu_bundle.py) -->
Measured at git `3c43e08`: `Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct`, Tesla T4, fp16, batch 1, drafter strategy `composite-g6-f2` (prompt lookup up to 6 tokens when the context has a match, else the model drafter drafts 2; requested `auto`), 10 prompts per set (chat template), 128 new tokens. One model pair, one GPU. Speedups are medians of per-prompt ratios within the same run. The headline baseline is HuggingFace `generate` with a static cache compiled the same way (CUDA graphs); eager HuggingFace and assisted generation are context. Cells are the stored JSON values. Acceptance is not a stored value: it is accepted / proposed draft tokens over the timed calls, computed from the stored `arm_report` step counts.

| dataset | mode | FlashSpec tok/s | compiled HF tok/s | **vs compiled HF** | eager HF tok/s | vs eager HF | vs HF assisted | acceptance | c (compiled) | greedy match vs compiled HF | greedy match vs eager HF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| humaneval | greedy | 84.15121158912345 | 50.84919550812026 | **1.6703915213699534** | 24.429024644223723 | 3.4316137147436416 | 3.6799192622210697 | 0.3775609756097561 | 0.47814633355546243 | 9/10 | 10/10 |
| humaneval | T=0.7 | 75.73985864164916 | 50.42612714053795 | **1.5026896386679338** | 24.497491036510233 | 3.066243803639478 | 3.9158860079861073 | 0.3888888888888889 | 0.48198575930759086 | n/a | n/a |
| mt_bench | greedy | 65.5665446517103 | 51.56689852526201 | **1.2745206859837361** | 24.358718936743088 | 2.6870360558531057 | 3.9853712833572263 | 0.2789115646258503 | 0.4786350054126751 | 8/10 | 8/10 |
| mt_bench | T=0.7 | 63.11715269934044 | 50.24734945285092 | **1.2630258753083248** | 24.371632087092596 | 2.5801491971418065 | 4.474878921310762 | 0.2970561998215879 | 0.4804902594066257 | n/a | n/a |

Greedy: compiled FlashSpec produced the same tokens as `hf_greedy_compiled`, token for token (per-prompt output hashes are in the JSON), on 9/10 humaneval prompts and 8/10 mt_bench prompts.
Cold start: the first compiled FlashSpec call in this run (mt_bench, greedy) took 223225.92456299992 ms including compilation and CUDA-graph capture.
<!-- END GENERATED: status -->

How to read it: each row is one prompt set and decoding mode; **vs compiled HF** is the median per-prompt speedup over HuggingFace `generate` compiled the same way (above 1 means FlashSpec was faster), and the last two columns count the prompts whose greedy tokens matched.

![Bar chart of tokens per second on a Tesla T4 for compiled FlashSpec, compiled HuggingFace generate and eager HuggingFace generate, on HumanEval and MT-Bench, greedy and T=0.7 (bundle 3c43e08, default settings). The exact values are in the table above.](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/throughput_3c43e08.png)

The chart is drawn from the same JSON field as the table by `scripts/readme_chart.py`; it adds no number of its own.

The full per-method table for the same run (eager FlashSpec and every HuggingFace baseline):

<!-- BEGIN GENERATED: gpu-results (scripts/ingest_gpu_bundle.py) -->
#### `Qwen/Qwen2.5-1.5B-Instruct` verified, `Qwen/Qwen2.5-0.5B-Instruct` drafting

GPU Tesla T4 · torch 2.11.0+cu130 · drafter strategy `composite-g6-f2` (prompt lookup up to 6 tokens when the context has a match, else the model drafter drafts 2; requested `auto`) · 20 result files · git `3c43e08`. Prompts: chat_template; n per row: 10. Cells are the stored JSON values, not rounded. Acceptance is not a stored value: it is accepted / proposed draft tokens over the timed calls, computed from the stored `arm_report` step counts.

| dataset | mode | method | tok/s median | tok/s p25 | tok/s p75 | acceptance | tokens / target fwd | target fwd / token | speedup vs plain HF (median) | speedup vs compiled plain HF (median) | speedup vs hf_assisted (median) | c (step profile) | greedy match | NaN/inf logits | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| humaneval | greedy | flashspec_ucb | 35.695688670935155 | 29.562405679239028 | 38.56945102463008 | 0.3759727626459144 | 2.471042471042471 | 0.4125 | 1.451343369376347 | 0.7066183063738005 | 1.5587746305864147 | 0.8669655800732762 | 10/10 | 0 | 3.9214916229248047 |
| humaneval | greedy | flashspec_compiled | 84.15121158912345 | 72.74887834456933 | 89.15351141840875 | 0.3775609756097561 | 2.4758220502901356 | 0.41171875 | 3.4316137147436416 | 1.6703915213699534 | 3.6799192622210697 | 0.5669970109881699 | 10/10 | n/a | 3.935239791870117 |
| humaneval | greedy | hf_greedy | 24.429024644223723 | 24.328362833297252 | 24.54774113031622 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9279322624206543 |
| humaneval | greedy | hf_greedy_compiled | 50.84919550812026 | 50.817140359183256 | 51.01212214615007 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | 9/10 | n/a | 3.9221267700195312 |
| humaneval | greedy | hf_assisted | 23.302891492129334 | 21.183524392621223 | 24.424857044996198 | n/a | 7.231638418079096 | 0.14609375 | n/a | n/a | n/a | n/a | 10/10 | 0 | 4.010546684265137 |
| humaneval | T=0.7 | flashspec_ucb | 28.096687974443004 | 25.489319336463833 | 37.08217864854619 | 0.3885480572597137 | 2.397003745318352 | 0.425 | 1.1429797583795867 | 0.5554736867969583 | 1.5315892372032842 | 0.8643673380226606 | n/a | 0 | 3.942927360534668 |
| humaneval | T=0.7 | flashspec_compiled | 75.73985864164916 | 66.31949556957646 | 86.96726957235924 | 0.3888888888888889 | 2.401500938086304 | 0.42421875 | 3.066243803639478 | 1.5026896386679338 | 3.9158860079861073 | 0.5533693858522074 | n/a | n/a | 3.943270206451416 |
| humaneval | T=0.7 | hf_sample | 24.497491036510233 | 24.352074445631605 | 24.59023809444537 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9425158500671387 |
| humaneval | T=0.7 | hf_sample_compiled | 50.42612714053795 | 50.38888740804433 | 50.596619340479606 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | 3.9360790252685547 |
| humaneval | T=0.7 | hf_assisted | 18.844078044479534 | 18.04090535750411 | 21.24417385471367 | n/a | 6.037735849056604 | 0.1734375 | n/a | n/a | n/a | n/a | n/a | 0 | 4.024676322937012 |
| mt_bench | greedy | flashspec_ucb | 24.83938465726492 | 22.961833475251858 | 27.021801893823184 | 0.2712363330529857 | 1.9968798751950079 | 0.50859375 | 1.0179685727219971 | 0.4828425422653919 | 1.5451784248202172 | 0.8613753375863016 | 9/10 | 0 | 3.9019665718078613 |
| mt_bench | greedy | flashspec_compiled | 65.5665446517103 | 59.93265391596037 | 71.89633242799991 | 0.2789115646258503 | 2.022116903633491 | 0.50234375 | 2.6870360558531057 | 1.2745206859837361 | 3.9853712833572263 | 0.5318390145420024 | 8/10 | n/a | 3.9116411209106445 |
| mt_bench | greedy | hf_greedy | 24.358718936743088 | 24.18066265396688 | 24.491882521750334 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9123830795288086 |
| mt_bench | greedy | hf_greedy_compiled | 51.56689852526201 | 51.44110439771817 | 51.6446975154745 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | 10/10 | n/a | 3.9070162773132324 |
| mt_bench | greedy | hf_assisted | 18.834736736908617 | 14.102116189597641 | 20.858267222930852 | n/a | 3.8905775075987843 | 0.26484375 | n/a | n/a | n/a | n/a | 10/10 | 0 | 3.982469081878662 |
| mt_bench | T=0.7 | flashspec_ucb | 25.26133407738635 | 23.020854972868243 | 28.43236543070436 | 0.2892670157068063 | 2.0611916264090175 | 0.49296875 | 1.0434498871962403 | 0.5027379844644687 | 1.784691816669535 | 0.8687020810505087 | n/a | 0 | 3.9193735122680664 |
| mt_bench | T=0.7 | flashspec_compiled | 63.11715269934044 | 59.5644483083085 | 69.07556580010277 | 0.2970561998215879 | 2.071197411003236 | 0.490625 | 2.5801491971418065 | 1.2630258753083248 | 4.474878921310762 | 0.5388284128435155 | n/a | n/a | 3.9206857681274414 |
| mt_bench | T=0.7 | hf_sample | 24.371632087092596 | 24.137529875735552 | 24.643657960930078 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9260549545288086 |
| mt_bench | T=0.7 | hf_sample_compiled | 50.24734945285092 | 50.13961412887575 | 50.398525858279534 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | 3.9206881523132324 |
| mt_bench | T=0.7 | hf_assisted | 14.341218875882541 | 9.085275562353358 | 18.19088437853941 | n/a | 3.9751552795031055 | 0.259375 | n/a | n/a | n/a | n/a | n/a | 0 | 4.005721092224121 |
- forward_cost (humaneval): c eager 0.8987917949296391, c compiled 0.48198575930759086
- forward_cost (humaneval): c eager 0.9153857271517092, c compiled 0.47814633355546243
- forward_cost (mt_bench): c eager 0.9028193999622582, c compiled 0.4786350054126751
- forward_cost (mt_bench): c eager 0.9178312220110205, c compiled 0.4804902594066257

Speedups are medians of per-prompt ratios against the method in the same run. NaN/inf counts come from one untimed probe generation per method on the first prompt.
<!-- END GENERATED: gpu-results -->

Provenance, the earlier bundles, the older 0.1.5 runs, kernel microbenchmarks and how a run is recorded are in [docs/benchmarks.md](https://github.com/Mattral/FlashSpec/blob/main/docs/benchmarks.md). Every number above is a stored JSON value under [benchmarks/results/gpu/3c43e08](https://github.com/Mattral/FlashSpec/tree/main/benchmarks/results/gpu/3c43e08). The 0.2.0 tables (model drafter, bundle `8d036d7`) are kept in the same folder layout.

## How it works

Each batch-1 step does four things:

1. The drafter proposes gamma tokens, with gamma forwards on its own KV cache (a lookup drafter runs no model).
2. The target runs one forward over the pending token and the gamma drafts.
3. The accept/reject test runs on the device: Algorithm 1 rejection sampling at temperature > 0, or the longest prefix that matches the target's argmax at temperature 0.
4. One device-to-host read returns the accept count and the next token, and both caches are cropped by bookkeeping, with no extra forward.

With `compile` on, both models run on static KV caches and the fixed-shape decode and verify calls replay as CUDA graphs. Diagrams, batching and the proof sketch: [docs/architecture.md](https://github.com/Mattral/FlashSpec/blob/main/docs/architecture.md).

## Correctness

- **Sampling is lossless.** Rejection sampling follows Algorithm 1 of Leviathan et al. (2023): a draft token `x` is kept with probability `min(1, p(x)/q(x))`, and a rejection samples from `max(0, p - q)`, normalised. The output then follows your model's distribution for any drafter. CI checks this with chi-squared tests on discrete tokens: [test_output_distribution.py](https://github.com/Mattral/FlashSpec/blob/main/tests/integration/test_output_distribution.py) for model drafters and [test_output_distribution_ngram.py](https://github.com/Mattral/FlashSpec/blob/main/tests/integration/test_output_distribution_ngram.py) for prompt lookup, which also shows that the checks fail on deliberately wrong residuals.
- **Greedy equals HuggingFace.** At temperature 0 the output is `model.generate(do_sample=False)` token for token. CI checks it in float32 on CPU ([test_greedy_gold.py](https://github.com/Mattral/FlashSpec/blob/main/tests/unit/test_greedy_gold.py)).
- **The fp16 caveat.** On a GPU in fp16, kernels of different shapes can round an exact tie between the top two tokens differently, so greedy output can differ from HuggingFace at such a tie. A T4 diagnostic found every path token-exact in fp32, and every fp16 divergence at an exact top-2 tie on one side ([diagnostic JSON](https://github.com/Mattral/FlashSpec/blob/main/benchmarks/results/diagnostics/width_gpu_t4_125d1a1.json)). Every greedy GPU result reports how many prompts matched. In the 0.3.0 release run the divergences behind 9/10 (HumanEval) and 8/10 (MT-Bench) are those same three ties: both sides' outputs are token-for-token those of the tie-checked comparison run (bundle `fdd3be9`).
- `variant="typical"` is lossy. `"rejection"` and `"greedy"` are the lossless variants.

## Status and limitations

FlashSpec is alpha, version 0.3.2.

- **Measured scope.** One model pair (`Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct`), one Tesla T4, fp16, batch 1, 10 MT-Bench and 10 HumanEval prompts, run from `main` at git `3c43e08` with the defaults `compile="auto"` and `drafter_strategy="auto"` (which ran `composite-g6-f2`). No other GPU, model pair or batch size has been measured. Every H100 number in the docs is a design target.
- **The first call compiles.** On CUDA the first `generate` compiles the forwards and captures CUDA graphs, which can take minutes; later calls reuse them. The tables exclude that call.
- **Batch > 1** uses the generic loop: each step forwards only the largest group of rows that share a context length.
- **Single device.** One process and one GPU or CPU. The server has no streaming, no paged KV cache and no continuous batching.
- **Windows and macOS.** `pip install flashspec` works, but Triton is Linux-only, so the compiled path is Linux + CUDA only.
- **Default drafter strategy.** `"auto"`, which runs `composite-g6-f2` with one model drafter at batch 1. Pass `drafter_strategy="bandit"` for the 0.2.0 behaviour.
- **Experimental: the bandit.** The UCB1/Thompson selector was measured choosing among seven arms once (Phase 6, bundle `3e38251`) and did not beat the best fixed arm on the mixed workload. See [docs/bandit.md](https://github.com/Mattral/FlashSpec/blob/main/docs/bandit.md) and the [CHANGELOG](https://github.com/Mattral/FlashSpec/blob/main/CHANGELOG.md).
- **Experimental: the kernels.** The Triton `verify_tokens` / `gather_accepted` kernels and the assembly residual are not called on the generation path; the accept/reject test runs in PyTorch. Measurements: [docs/kernels.md](https://github.com/Mattral/FlashSpec/blob/main/docs/kernels.md).
- Waiting on a GPU or an upload: [docs/postponed.md](https://github.com/Mattral/FlashSpec/blob/main/docs/postponed.md).

### Future work

Not scheduled and not implemented:

- **A contextual bandit, run once.** The Phase 6 bandit had no context, and the best fixed arm changed between consecutive prompts of the mixed workload more often than not. A pre-registered design: the context is the next lookup's match length bucket ({0, 1-2, 3-4, >= 5}), which the step's single device read already carries; with no match the arms are Qwen at gamma 2, 4 and 6, otherwise n-gram and Qwen at gamma 2, 4 and 6; the value per (context, arm) is total emitted tokens / total seconds, not a mean of per-step ratios; UCB1, normalised, verify width 7. Its exit criteria would be fixed before the run (beat `composite-g4` and every fixed arm on the mixed workload in both modes, within 5% of each set's best fixed arm, faster than compiled HuggingFace everywhere, no new greedy mismatches), and a miss would be recorded without tuning or a second attempt. The oracle ceiling estimated from the Phase 6 bundle is small, so near parity with the composite strategy is the expected outcome.
- A larger target (Qwen2.5-3B-Instruct with the 0.5B drafter on the T4), the batch > 1 path, and draft log-probs kept in the model dtype.

## Citing

```bibtex
@misc{myet2026flashspec,
  title = {{FlashSpec}: Adaptive Speculative Decoding with Online Bandit
             Draft Selection and {Triton}-Optimised Verification},
  author = {Myet, Min Htet},
  year = {2026},
  month = jun,
  howpublished = {Zenodo preprint},
  doi = {10.5281/zenodo.20766119},
  url = {https://doi.org/10.5281/zenodo.20766119},
}
```

**Preprint DOI**: [10.5281/zenodo.20766119](https://doi.org/10.5281/zenodo.20766119) (June 2026). That record is the citable archive. [`paper/flashspec.tex`](https://github.com/Mattral/FlashSpec/blob/main/paper/flashspec.tex) is a later draft (October 2026, software 0.2.0) and has not been deposited. Machine-readable metadata is in [`CITATION.cff`](https://github.com/Mattral/FlashSpec/blob/main/CITATION.cff).

## Contributing

See [CONTRIBUTING.md](https://github.com/Mattral/FlashSpec/blob/main/CONTRIBUTING.md). `make test` runs the unit and integration suites on CPU, with no GPU needed. Test commands, the repository layout and links to the papers and notebooks are in [docs/development.md](https://github.com/Mattral/FlashSpec/blob/main/docs/development.md). Changes are listed in the [CHANGELOG](https://github.com/Mattral/FlashSpec/blob/main/CHANGELOG.md). An index for AI agents is in [llms.txt](https://github.com/Mattral/FlashSpec/blob/main/llms.txt).

## License

Apache 2.0. See [LICENSE](https://github.com/Mattral/FlashSpec/blob/main/LICENSE).
