# FlashSpec

Lossless speculative decoding for HuggingFace causal language models, with a static-cache CUDA-graph fast path.

[![PyPI](https://img.shields.io/pypi/v/flashspec)](https://pypi.org/project/flashspec/)
[![Python](https://img.shields.io/pypi/pyversions/flashspec)](https://pypi.org/project/flashspec/)
[![CI](https://github.com/Mattral/FlashSpec/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/FlashSpec/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://flashspec.readthedocs.io)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20766119-blue)](https://doi.org/10.5281/zenodo.20766119)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Mattral/FlashSpec/blob/main/LICENSE)

## What it is

FlashSpec wraps an existing HuggingFace causal LM and a smaller drafter that shares its tokenizer: the drafter proposes tokens, your model checks them in one forward pass, and the output is your model's own: the same distribution when sampling, and the same tokens in greedy mode up to fp16 ties ([correctness](https://github.com/Mattral/FlashSpec#correctness)). On one Tesla T4, with `Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct` at batch 1, it was faster than HuggingFace `generate` compiled the same way in every measured cell, by at least 1.3x on HumanEval and by less on MT-Bench ([results](https://github.com/Mattral/FlashSpec#results)). That is the whole measured scope (one model pair, one GPU, batch 1), and FlashSpec is a single-device library with a small HTTP server, not a serving stack.

## When to use it

Use it when:

- you run a HuggingFace causal LM on a CUDA GPU, one request at a time (batch 1);
- you have a smaller drafter with the same tokenizer, or your outputs copy from the prompt (code edits, quoting), where the built-in prompt-lookup drafter needs no second model;
- you need your model's exact output, not an approximation of it.

Not yet:

- **CPU.** Both recorded CPU end-to-end runs were slower than plain decoding ([history](https://github.com/Mattral/FlashSpec/blob/main/docs/benchmarks.md#historical-and-microbenchmark-results)).
- **Batch > 1.** Larger batches use a generic loop that reads the device several times per step and has not been measured.
- **First-call latency.** On CUDA the first `generate` compiles and captures CUDA graphs, which can take minutes. `compile=False` skips that, but the eager path was slower than plain HuggingFace `generate` in 3 of the 4 measured cells.
- **Other GPUs and model pairs.** They have not been measured.

## Install

```bash
pip install flashspec
```

| extra | adds | when you want it |
|---|---|---|
| `flashspec[gpu]` | Triton (Linux only) | The compiled path needs Triton; Linux CUDA builds of PyTorch already ship it. Also enables the experimental Triton kernels. |
| `flashspec[serve]` | FastAPI, uvicorn, httpx | The OpenAI-compatible server. Also `pip install accelerate`: the server loads checkpoints with `device_map`. |
| `flashspec[asm]` | `flashspec-asm-kernel` (Linux x86-64) | An experimental CPU residual kernel. It does not change end-to-end speed. |

You need Python 3.11 or later, PyTorch 2.2 or later and transformers 4.40 or later. Installing from source, Docker, Windows and macOS notes, and the full requirements table are in [docs/install.md](https://github.com/Mattral/FlashSpec/blob/main/docs/install.md).

## Quickstart

This loads the measured model pair, decodes greedily, and prints the text and the speed. Without CUDA it falls back to CPU and float32, which is slow.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashspec import BanditConfig, FlashSpecConfig, SamplingConfig, SpeculativeEngine
from flashspec.bandit import UCB1Selector
from flashspec.engine import HuggingFaceDrafter, TargetModel
target_id, drafter_id = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"  # same tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
load = lambda name: AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
target = TargetModel(target_id, device, dtype, model=load(target_id))
drafter = HuggingFaceDrafter(drafter_id, device, dtype, model=load(drafter_id))
config = FlashSpecConfig(device=str(device), bandit=BanditConfig(n_arms=1),
                         sampling=SamplingConfig(temperature=0.0))  # greedy
engine = SpeculativeEngine(config, [drafter], target, UCB1Selector(n_arms=1))
tokenizer = AutoTokenizer.from_pretrained(target_id)
prompt = tokenizer("def fibonacci(n):", return_tensors="pt").input_ids.to(device)
engine.generate(prompt, max_new_tokens=128)  # the first call on CUDA compiles (can take minutes)
result = engine.generate(prompt, max_new_tokens=128)
print(tokenizer.decode(result.output_ids[0, : result.n_tokens_generated], skip_special_tokens=True))
print(f"{result.tokens_per_second:.1f} tok/s, acceptance {result.acceptance_rate:.2f}")
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
| `"bandit"` | one arm per drafter at `sampling.gamma`, chosen online by the selector you pass (experimental) | one or more drafters |
| `"composite-g4"` | prompt lookup, up to 4 tokens, when the context has a match; else the model drafter drafts 4 | a model drafter and an `NgramDrafter` |
| `"composite-g6-f2"` | prompt lookup, up to 6 tokens, when the context has a match; else the model drafter drafts 2 | a model drafter and an `NgramDrafter` |

A composite strategy is one fixed arm (`bandit.n_arms` defaults to 1), runs at batch 1 with `variant` `"rejection"` or `"greedy"`, and verifies at width `max gamma + 1`. Lookup steps use Algorithm 1 with a one-hot draft distribution, so it is lossless; the chi-squared tests cover separate lookup and fallback gammas. Which composite becomes the default is decided by the pre-registered comparison in `benchmarks/configs/qwen_t4_composite.yaml` (applied by `scripts/composite_report.py`).

```python
from flashspec.engine import ArmSpec, NgramDrafter

lookup = SpeculativeEngine(config, [NgramDrafter()], target, UCB1Selector(n_arms=1))
composite = SpeculativeEngine(
    config, [drafter, NgramDrafter()], target, UCB1Selector(n_arms=1),
    arms=[ArmSpec(drafter=1, gamma=4, fallback=0)],  # drafter 1 = lookup, fallback 0 = model
)
result = composite.generate(prompt, max_new_tokens=64)
print([step.drafter for step in result.arm_steps])  # which drafter proposed each step
```

Which is fastest depends on the prompts. In the Phase 6 T4 run (greedy, bundle `3e38251`), the fastest fixed choice was the composite at gamma 4 on HumanEval and on the mixed workload, the model drafter at gamma 2 on MT-Bench, and prompt lookup at gamma 6 on code edits ([values](https://github.com/Mattral/FlashSpec/tree/main/benchmarks/results/gpu/3e38251)). Several arms plus a UCB1 or Thompson selector let a bandit choose online; that is experimental, and in the same run it did not beat the best fixed arm ([bandit](https://github.com/Mattral/FlashSpec/blob/main/docs/bandit.md)).

### Greedy or sampling

```python
greedy = SamplingConfig(temperature=0.0)            # same tokens as model.generate(do_sample=False)
sampled = SamplingConfig(temperature=0.7, seed=0)   # Algorithm 1: same distribution as the target
drafter.set_temperature(0.7)                        # draft at the target's temperature
config = FlashSpecConfig(device=str(device), bandit=BanditConfig(n_arms=1), sampling=sampled)
engine = SpeculativeEngine(config, [drafter], target, UCB1Selector(n_arms=1))
result = engine.generate(prompt, max_new_tokens=64)
```

The default is `temperature=1.0`, which samples. `temperature=0.0` selects greedy. `seed` makes one call reproducible; `seed=None` uses the global RNG. `variant="typical"` is lossy and is not on the fast path.

### Turn compile off

```python
config = FlashSpecConfig(device=str(device), compile=False, bandit=BanditConfig(n_arms=1),
                         sampling=SamplingConfig(temperature=0.0))
engine = SpeculativeEngine(config, [drafter], target, UCB1Selector(n_arms=1))
print(engine.compiled, "|", engine.compile_decision.reason)
```

The default `compile="auto"` turns the compiled path on when the device is CUDA and Triton is installed. The engine logs one line saying which path runs and why. Set `compile=False` before a compiled engine has run on the same `TargetModel` and drafter: the compiled path moves those wrappers onto static caches, and they stay there. The server takes `--compile off`.

### Run the OpenAI-compatible server

```bash
pip install "flashspec[serve]" accelerate
python -m flashspec.serve --target Qwen/Qwen2.5-1.5B-Instruct --drafter Qwen/Qwen2.5-0.5B-Instruct \
    --device cuda:0 --dtype float16 --temperature 0
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

own = SpeculativeEngine(config, [RepeatLast(target.vocab_size)], target, UCB1Selector(n_arms=1))
result = own.generate(prompt, max_new_tokens=32)
```

The proposals must be draws from the distribution you return. Algorithm 1 is then exact for any drafter, and a poor drafter only lowers the acceptance rate. To build a drafter by name with `load_drafters`, decorate its class with `@flashspec.register("name")`; the constructor then receives `device`, `dtype` and `temperature`.

## API at a glance

The compute dtype is the one the model wrappers load with or are given; `FlashSpecConfig.dtype` is recorded with the config and not applied.

<!-- BEGIN GENERATED: api (scripts/readme_api.py) -->
Generated from the code by `scripts/readme_api.py`; a test fails when it is stale.

| API | What it does | Key arguments, with defaults |
|---|---|---|
| `flashspec.FlashSpecConfig` | Top-level FlashSpec runtime configuration. | fields: `drafter_name: str = "llama3-1b"`, `target_name: str = "meta-llama/Llama-3-8B-Instruct"`, `device: str = "cuda:0"`, `dtype: "float32" \| "bfloat16" \| "float16" = "bfloat16"`, `max_new_tokens: int = 512`, `eos_token_ids: tuple[int, ...] = ()`, `drafter_strategy: "bandit" \| "composite-g4" \| "composite-g6-f2" = "bandit"`, `compile: bool \| "auto" = "auto"`, `compile_max_cache_len: int = 1024`, `compile_fixed_shapes: bool = True`, `bandit: BanditConfig = BanditConfig()`, `sampling: SamplingConfig = SamplingConfig()`, `metrics: MetricsConfig = MetricsConfig()` |
| `flashspec.SamplingConfig` | Configuration for speculative sampling. | fields: `gamma: int = 4`, `temperature: float = 1.0`, `top_p: float = 1.0`, `seed: int \| None = None`, `variant: "rejection" \| "typical" \| "greedy" = "rejection"` |
| `flashspec.BanditConfig` | Configuration for the online bandit draft selector. | fields: `strategy: "ucb1" \| "thompson" \| "oracle" = "ucb1"`, `window_size: int = 500`, `n_arms: int = 2`, `ucb_exploration_constant: float = 1.0`, `thompson_prior_alpha: float = 1.0`, `thompson_prior_beta: float = 1.0` |
| `flashspec.SpeculativeEngine` | Adaptive speculative-decoding inference engine. | arguments: `config: FlashSpecConfig`, `drafters: list[DraftModel]`, `target: TargetModel`, `bandit: DraftSelector`, `arms: Sequence[ArmSpec] \| None = None` |
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
- Drafter strategy: `SpeculativeEngine(..., arms=None)` makes one arm per drafter at `sampling.gamma`, and the selector you pass chooses between them. With one drafter that is a fixed drafter.
- `BanditConfig.n_arms = 2`: it must equal the number of arms, so pass `BanditConfig(n_arms=1)` for one drafter.
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
| `--drafter-strategy` | `"bandit"` | "bandit", "composite-g4", "composite-g6-f2" |
| `--strategy` | `"ucb1"` | "ucb1", "thompson" |
| `--dtype` | `"float32"` | "float32", "bfloat16", "float16" |
| `--compile` | `"auto"` | "auto", "on", "off" |
<!-- END GENERATED: api -->

The full reference is generated from the docstrings: [API reference](https://flashspec.readthedocs.io/en/latest/api/).

## Results

<!-- BEGIN GENERATED: status (scripts/ingest_gpu_bundle.py) -->
Measured at git `8d036d7`: `Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct`, Tesla T4, fp16, batch 1, gamma 4, 10 prompts per set (chat template), 128 new tokens. One model pair, one GPU. Speedups are medians of per-prompt ratios within the same run. The headline baseline is HuggingFace `generate` with a static cache compiled the same way (CUDA graphs); eager HuggingFace and assisted generation are context. Cells are the stored JSON values.

| dataset | mode | FlashSpec tok/s | compiled HF tok/s | **vs compiled HF** | eager HF tok/s | vs eager HF | vs HF assisted | acceptance | c (compiled) | greedy match vs compiled HF | greedy match vs eager HF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| humaneval | greedy | 73.49117484423093 | 49.27609336228632 | **1.4929247183661758** | 25.272922944874917 | 2.868400153448608 | 2.968039793678412 | 0.7674472295086122 | 0.47651099333876806 | 10/10 | 9/10 |
| humaneval | T=0.7 | 65.40179447333742 | 49.24821559481592 | **1.326870202978973** | 25.444714453071143 | 2.562178271321323 | 3.06365302315802 | 0.7018560004266362 | 0.4770852877691722 | n/a | n/a |
| mt_bench | greedy | 55.69633073174627 | 49.99045416467851 | **1.1161741322307486** | 25.20267917967235 | 2.2281081944682457 | 3.0887126152344058 | 0.5310738087770345 | 0.47418886402988414 | 10/10 | 10/10 |
| mt_bench | T=0.7 | 53.447483345537975 | 49.189297069159096 | **1.0837177249808096** | 25.344508384460696 | 2.119677479254399 | 3.3014455192389267 | 0.5374968568894605 | 0.47562151975963995 | n/a | n/a |

Greedy: compiled FlashSpec produced the same tokens as `hf_greedy_compiled`, token for token (per-prompt output hashes are in the JSON), on 10/10 humaneval prompts and 10/10 mt_bench prompts.
Cold start: not recorded in this bundle. The first compiled call compiles the model and captures CUDA graphs, which can take minutes; the timed numbers above exclude it (warmup prompts absorb it).
<!-- END GENERATED: status -->

How to read it: each row is one prompt set and decoding mode; **vs compiled HF** is the median per-prompt speedup over HuggingFace `generate` compiled the same way (above 1 means FlashSpec was faster), and the last two columns count the prompts whose greedy tokens matched.

The full per-method table for the same run (eager FlashSpec and every HuggingFace baseline):

<!-- BEGIN GENERATED: gpu-results (scripts/ingest_gpu_bundle.py) -->
#### `Qwen/Qwen2.5-1.5B-Instruct` verified, `Qwen/Qwen2.5-0.5B-Instruct` drafting

GPU Tesla T4 · torch 2.11.0+cu130 · gamma 4 · 20 result files · git `8d036d7`. Prompts: chat_template; n per row: 10. Cells are the stored JSON values, not rounded.

| dataset | mode | method | tok/s median | tok/s p25 | tok/s p75 | acceptance | tokens / target fwd | target fwd / token | speedup vs plain HF (median) | speedup vs compiled plain HF (median) | speedup vs hf_assisted (median) | c (step profile) | greedy match | NaN/inf logits | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| humaneval | greedy | flashspec_ucb | 26.02536363267037 | 24.113137938821488 | 26.416962512055154 | 0.7748324561339393 | 4.19672131147541 | 0.24609375 | 1.0193827466008485 | 0.5282968591639221 | 1.0318343857730912 | 0.845041661430388 | 10/10 | 0 | 3.9210314750671387 |
| humaneval | greedy | flashspec_compiled | 73.49117484423093 | 67.66749962983283 | 75.22223233061746 | 0.7674472295086122 | 4.169381107491857 | 0.24765625 | 2.868400153448608 | 1.4929247183661758 | 2.968039793678412 | 0.4937086544815969 | 9/10 | n/a | 3.9358386993408203 |
| humaneval | greedy | hf_greedy | 25.272922944874917 | 25.064466964565117 | 25.500231713400083 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9280996322631836 |
| humaneval | greedy | hf_greedy_compiled | 49.27609336228632 | 49.1847500987219 | 49.34777025069719 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | 9/10 | n/a | 3.922210216522217 |
| humaneval | greedy | hf_assisted | 24.19664417567833 | 22.245649238351376 | 25.200750728941774 | n/a | 7.231638418079096 | 0.14609375 | n/a | n/a | n/a | n/a | 10/10 | 0 | 4.010560035705566 |
| humaneval | T=0.7 | flashspec_ucb | 23.054835447878503 | 21.110460201553547 | 24.779995491896514 | 0.6882722339966006 | 3.710144927536232 | 0.27734375 | 0.9109461711070447 | 0.46804709580075965 | 1.0736463665588767 | 0.8629833155074336 | n/a | 0 | 3.9422059059143066 |
| humaneval | T=0.7 | flashspec_compiled | 65.40179447333742 | 59.35656427020519 | 68.5928206976523 | 0.7018560004266362 | 3.764705882352941 | 0.2734375 | 2.562178271321323 | 1.326870202978973 | 3.06365302315802 | 0.49779909081468227 | n/a | n/a | 3.9422059059143066 |
| humaneval | T=0.7 | hf_sample | 25.444714453071143 | 25.347879588597422 | 25.543325368357564 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9432334899902344 |
| humaneval | T=0.7 | hf_sample_compiled | 49.24821559481592 | 49.19260891088411 | 49.33891386116446 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | 3.936194896697998 |
| humaneval | T=0.7 | hf_assisted | 20.25776424898342 | 19.350477197434962 | 22.45474429590343 | n/a | 6.037735849056604 | 0.1734375 | n/a | n/a | n/a | n/a | n/a | 0 | 4.024886608123779 |
| mt_bench | greedy | flashspec_ucb | 19.572806929034726 | 18.512325739597685 | 20.655499375866352 | 0.5868378012374362 | 3.2653061224489797 | 0.3140625 | 0.7776848037196118 | 0.3925474950495035 | 1.1001251488328485 | 0.8672852493814422 | 9/10 | 0 | 3.902125835418701 |
| mt_bench | greedy | flashspec_compiled | 55.69633073174627 | 52.71416447738182 | 57.00448545307138 | 0.5310738087770345 | 3.18407960199005 | 0.321875 | 2.2281081944682457 | 1.1161741322307486 | 3.0887126152344058 | 0.48626613558395965 | 10/10 | n/a | 3.9118690490722656 |
| mt_bench | greedy | hf_greedy | 25.20267917967235 | 25.112631097015935 | 25.30654132302575 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9123334884643555 |
| mt_bench | greedy | hf_greedy_compiled | 49.99045416467851 | 49.59376567557413 | 50.24789576812648 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | 10/10 | n/a | 3.9069666862487793 |
| mt_bench | greedy | hf_assisted | 19.849014490485814 | 15.136578031686579 | 22.22945913877053 | n/a | 3.8905775075987843 | 0.26484375 | n/a | n/a | n/a | n/a | 10/10 | 0 | 3.9825186729431152 |
| mt_bench | T=0.7 | flashspec_ucb | 20.18121040233048 | 18.115274461600237 | 21.17643975210253 | 0.5437414602239845 | 3.3333333333333335 | 0.3078125 | 0.7952682871520586 | 0.41020068096677864 | 1.234533396317398 | 0.8612503546546161 | n/a | 0 | 3.9191641807556152 |
| mt_bench | T=0.7 | flashspec_compiled | 53.447483345537975 | 50.30655289847409 | 59.43461086804307 | 0.5374968568894605 | 3.2989690721649483 | 0.3109375 | 2.119677479254399 | 1.0837177249808096 | 3.3014455192389267 | 0.5416823432691761 | n/a | n/a | 3.923170566558838 |
| mt_bench | T=0.7 | hf_sample | 25.344508384460696 | 25.114667710180996 | 25.389395380475488 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | 0 | 3.9262752532958984 |
| mt_bench | T=0.7 | hf_sample_compiled | 49.189297069159096 | 49.090311507338804 | 49.27322962156904 | n/a | 1.0078740157480315 | 1.0 | n/a | n/a | n/a | n/a | n/a | n/a | 3.9209084510803223 |
| mt_bench | T=0.7 | hf_assisted | 15.343815220295882 | 9.802755592886559 | 19.533044949645106 | n/a | 3.9751552795031055 | 0.259375 | n/a | n/a | n/a | n/a | n/a | 0 | 4.005769729614258 |
- forward_cost (humaneval): c eager 0.8840264407247382, c compiled 0.47651099333876806
- forward_cost (humaneval): c eager 0.8922046962713643, c compiled 0.4770852877691722
- forward_cost (mt_bench): c eager 0.8978235042101232, c compiled 0.47562151975963995
- forward_cost (mt_bench): c eager 0.9040499147358744, c compiled 0.47418886402988414

Speedups are medians of per-prompt ratios against the method in the same run. NaN/inf counts come from one untimed probe generation per method on the first prompt.
<!-- END GENERATED: gpu-results -->

Provenance, the earlier bundles, the older 0.1.5 runs, kernel microbenchmarks and how a run is recorded are in [docs/benchmarks.md](https://github.com/Mattral/FlashSpec/blob/main/docs/benchmarks.md). Every number above is a stored JSON value under [benchmarks/results/gpu/8d036d7](https://github.com/Mattral/FlashSpec/tree/main/benchmarks/results/gpu/8d036d7).

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
- **The fp16 caveat.** On a GPU in fp16, kernels of different shapes can round an exact tie between the top two tokens differently, so greedy output can differ from HuggingFace at such a tie. A T4 diagnostic found every path token-exact in fp32, and every fp16 divergence at an exact top-2 tie on one side ([diagnostic JSON](https://github.com/Mattral/FlashSpec/blob/main/benchmarks/results/diagnostics/width_gpu_t4_125d1a1.json)). Every greedy GPU result reports how many prompts matched.
- `variant="typical"` is lossy. `"rejection"` and `"greedy"` are the lossless variants.

## Status and limitations

FlashSpec is alpha; the release on PyPI is 0.2.0.

- **Measured scope.** One model pair (`Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct`), one Tesla T4, fp16, batch 1, 10 MT-Bench and 10 HumanEval prompts, run from `main` at git `8d036d7` with the default `compile="auto"`. No other GPU, model pair or batch size has been measured. Every H100 number in the docs is a design target.
- **The first call compiles.** On CUDA the first `generate` compiles the forwards and captures CUDA graphs, which can take minutes; later calls reuse them. The tables exclude that call.
- **Batch > 1** uses the generic loop: each step forwards only the largest group of rows that share a context length.
- **Single device.** One process and one GPU or CPU. The server has no streaming, no paged KV cache and no continuous batching.
- **Windows and macOS.** `pip install flashspec` works, but Triton is Linux-only, so the compiled path is Linux + CUDA only.
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
