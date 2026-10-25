<h1 align="center">⚡ FlashSpec</h1>

<p align="center">
  <strong>Lossless speculative decoding for HuggingFace causal language models</strong><br>
  Algorithm 1 rejection sampling and exact greedy · static KV cache + CUDA graphs on CUDA · experimental bandit and kernels
</p>

<p align="center">
  
  <a href="https://pypi.org/project/flashspec/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/flashspec?style=for-the-badge&logo=pypi&logoColor=white&color=7C3AED"></a>
  <a href="https://pypi.org/project/flashspec/">
    <img alt="Python" src="https://img.shields.io/pypi/pyversions/flashspec?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/flashspec/">
    <img alt="Downloads" src="https://img.shields.io/pypi/dm/flashspec?style=for-the-badge&color=A78BFA"></a>
  <a href="https://flashspec.readthedocs.io">
    <img alt="Docs" src="https://img.shields.io/badge/docs-readthedocs-22C55E?style=for-the-badge&logo=readthedocs&logoColor=white"></a>
  <a href="https://doi.org/10.5281/zenodo.20766119">
    <img alt="DOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20766119-7C3AED?style=for-the-badge"></a>
  <a href="./CITATION.cff">
    <img alt="Cite" src="https://img.shields.io/badge/cite-CITATION.cff-brightgreen?style=for-the-badge"></a>
  <a href="./LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-A78BFA?style=for-the-badge"></a>
  <a href="https://github.com/Mattral/FlashSpec/tree/main/notebooks">
    <img alt="Open Notebooks in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="height:28px;">
  </a>

</p>

---

**Project status: alpha, version 0.2.0.**

**What was measured, and only this:** one model pair
(`Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct`),
one Tesla T4, fp16, batch 1, 10 MT-Bench and 10 HumanEval prompts, run
from `main` at git `8d036d7` with the default `FlashSpecConfig(compile="auto")`,
which turned the compiled path (static KV cache and CUDA graphs) on. On
that setup FlashSpec was faster than HuggingFace `generate` compiled the
same way in every cell of the table below: by at least 1.3x on both
HumanEval cells, and by less on MT-Bench. With the eager path
(`compile=False`) it was slower than plain HuggingFace `generate` in 3 of
the 4 cells of the same run (generated table further down). No other GPU,
model pair or batch size has been measured. Every H100 number in the docs
is a design target.

- **The first call compiles.** On CUDA with Triton, the default
  `compile="auto"` compiles the drafter and target forwards and captures
  CUDA graphs on the first `generate`, which can take minutes. Later calls
  reuse them. The tables above and below exclude that first call (warmup
  prompts absorb it). To skip compilation, pass
  `FlashSpecConfig(compile=False)` (the eager path, slower here) or
  `--compile off` to the server. The engine logs one line saying whether
  the compiled path is on and why.

- **Correctness** is checked in CI on CPU: a chi-squared test that emitted
  tokens follow the target when draft and target differ, and a
  token-for-token comparison of greedy output with
  `generate(do_sample=False)`. On the GPU in fp16, kernels for different
  shapes can round differently, so each greedy GPU run reports how many
  prompts matched (last columns of the table).
- **The bandit is experimental.** In the Phase 6 run on one T4 (bundle
  `3e38251`, see the CHANGELOG) it chose among seven arms and did not beat
  the best fixed arm on the mixed workload. The composite drafter
  strategies are the candidates for the 0.3.0 default; see
  [Drafter strategies](#drafter-strategies).
- **The Triton and assembly kernels are experimental** and not on the
  generation critical path; see [Experimental kernels](#experimental-kernels-not-on-the-critical-path).
- `sampling.variant="typical"` is **lossy**; `"rejection"` and `"greedy"`
  are the lossless variants.

This is a single-device library plus a small HTTP server, not a
production serving stack.

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

---

## What is FlashSpec?

Speculative decoding lets a small "draft" model propose γ tokens that a
larger "target" model checks in one forward pass. FlashSpec's batch-1
step is:

1. the drafter proposes γ tokens (γ incremental forwards on its KV cache);
2. the target runs **one** forward over the pending token(s) plus the γ
   drafts;
3. the accept/reject test runs on the device in PyTorch: Leviathan et al.
   Algorithm 1 rejection sampling at temperature > 0, or the longest
   prefix that matches the target's argmax at temperature 0 (greedy);
4. one device-to-host read per step returns the accept count and the next
   token, and both caches are cropped by integer bookkeeping.

On CUDA with Triton available, `FlashSpecConfig.compile` runs that loop on
static KV caches and replays the fixed-shape drafter decode and target
verify calls as CUDA graphs (`torch.compile(mode="reduce-overhead")`).

Two parts are experimental. The UCB1/Thompson selector can choose between
arms (a drafter and a gamma each), but in the one run where it did, it
lost to the best fixed arm. The Triton
verification kernels and the optional assembly residual are not called on
the generation path (see
[Experimental kernels](#experimental-kernels-not-on-the-critical-path)).

On finite probabilities the output distribution follows Leviathan et al.,
2023, Theorem 1. The CI check is a chi-squared test on discrete token ids.
`tests/integration/test_output_distribution.py` uses fixed pairs with
draft distribution `q` different from target distribution `p`, at gamma
1 and gamma 4. It compares the first emitted token to `p`, and, when
gamma is 4, the second emitted token as well. N is 30,000 and the
vocabulary is 6, so each expected count stays above 5. The critical
value is the chi-square 1% point for 5 degrees of freedom, 15.086. The
same check runs on the PyTorch residual and on a kernel that implements
the same formula. An engine test uses a toy target whose next token
depends on the previous token and compares the first two generated
tokens with sampling from that target. `tests/integration/test_e2e_sampling.py`
is the gamma 1 case where the draft equals the target, so every draft
token is accepted. It does not exercise the residual. A failing
chi-squared test fails the suite.

---

## Quickstart

```bash
git clone https://github.com/Mattral/FlashSpec && cd FlashSpec

# Linux + CUDA (accelerated Triton kernels):
pip install -e ".[dev,gpu]"

# Windows / macOS (pure-PyTorch reference kernels, numerically identical):
pip install -e ".[dev]"

# Optional CPU residual kernel (Linux x86-64, flashspec-asm-kernel>=0.1.2;
# other platforms install nothing):
pip install -e ".[asm]"

# Optional single-device HTTP server:
pip install -e ".[serve]"
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashspec import FlashSpecConfig, BanditConfig, SamplingConfig, TRITON_AVAILABLE
from flashspec.bandit import UCB1Selector
from flashspec.engine import SpeculativeEngine, TargetModel, load_drafters

print(f"Triton kernels available: {TRITON_AVAILABLE}")

# Public checkpoints. Both report vocab_size 32000.
# Llama-3 reports vocab_size 128256. Do not pair it with this drafter.
drafter_id = "Felladrin/Llama-160M-Chat-v1"
target_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

config = FlashSpecConfig(
    device="cpu",
    dtype="float32",
    drafter_name=drafter_id,
    target_name=target_id,
    bandit=BanditConfig(n_arms=1, strategy="ucb1"),
    sampling=SamplingConfig(gamma=4, temperature=1.0),
    max_new_tokens=16,
)
# TargetModel(hub_id, device) passes device_map and needs accelerate,
# which this package does not depend on. Load the weights here and inject
# them, which is what benchmarks/harness.py does.
drafter_module = AutoModelForCausalLM.from_pretrained(
    drafter_id, torch_dtype=torch.float32
).to(config.device)
target_module = AutoModelForCausalLM.from_pretrained(
    target_id, torch_dtype=torch.float32
).to(config.device)
drafter_module.eval()
target_module.eval()
drafters = load_drafters(
    [drafter_id],
    device=config.device,
    dtype=torch.float32,
    temperature=config.sampling.temperature,
    models={drafter_id: drafter_module},
)
target = TargetModel(
    target_id,
    torch.device(config.device),
    torch.float32,
    model=target_module,
)
engine = SpeculativeEngine(
    config,
    drafters,
    target,
    UCB1Selector(n_arms=1, window_size=config.bandit.window_size),
)
tokenizer = AutoTokenizer.from_pretrained(target_id)
prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids
result = engine.generate(prompt, max_new_tokens=16)
new_ids = result.output_ids[0]
new_ids = new_ids[new_ids != -1]
print(tokenizer.decode(new_ids, skip_special_tokens=True))
```

Both checkpoints are public. `benchmarks/harness.py` refuses a drafter and a target whose vocabulary sizes differ and writes no result file. `SpeculativeEngine` itself does not apply that check. `notebooks/01_quickstart.py` is a separate CPU toy: the names are `"toy"`, the log-probs are random, and it does not download these checkpoints. With one drafter the bandit cannot adapt, and the loader says so.

### Drafter strategies

`FlashSpecConfig.drafter_strategy` chooses the arms the engine builds when
it is not given `arms=`. The server takes the same names as
`--drafter-strategy`.

| strategy | what drafts each step | drafters to pass |
|---|---|---|
| `"bandit"` (default) | one arm per drafter at `sampling.gamma`, chosen online by UCB1 or Thompson (experimental) | one or more model drafters |
| `"composite-g4"` | prompt lookup, up to 4 tokens, when the context has a match; otherwise the model drafter drafts 4 | one model drafter and an `NgramDrafter` |
| `"composite-g6-f2"` | prompt lookup, up to 6 tokens, when the context has a match; otherwise the model drafter drafts 2 | one model drafter and an `NgramDrafter` |

```python
# Continuing the quickstart above: the same drafter and target, plus prompt lookup.
from flashspec.engine import NgramDrafter

composite = FlashSpecConfig(
    device="cpu",
    dtype="float32",
    max_new_tokens=16,
    drafter_strategy="composite-g4",  # one arm: bandit.n_arms defaults to 1
    sampling=SamplingConfig(temperature=1.0),
)
engine = SpeculativeEngine(composite, [*drafters, NgramDrafter()], target, UCB1Selector(n_arms=1))
```

```bash
python -m flashspec.serve --target Qwen/Qwen2.5-1.5B-Instruct \
    --drafter Qwen/Qwen2.5-0.5B-Instruct --drafter-strategy composite-g4 --device cuda:0
```

A composite strategy is one fixed arm, so the bandit has nothing to choose.
It runs at batch 1 with `variant` `"rejection"` or `"greedy"` (the server
refuses `--batch-size` above 1 with it) and verifies at width
`max gamma + 1`: 5 for `composite-g4`, 7 for `composite-g6-f2`. Lookup
steps use Algorithm 1 with a one-hot draft distribution, so both are
lossless; the chi-squared tests cover separate lookup and fallback gammas.

**The default is still `"bandit"`, as in 0.2.0, until a measurement picks
one.** `benchmarks/configs/qwen_t4_composite.yaml` runs both composite
variants against compiled HuggingFace on the Phase 6 prompt sets (T4,
notebook cell 3f). Its rule was fixed before the run: the variant with the
higher mixed-workload greedy throughput is picked, and it becomes the
default only if it beats compiled HuggingFace on every prompt set in greedy
and T=0.7, and its greedy output matches compiled HuggingFace except at
fp16 exact ties. `scripts/composite_report.py` applies the rule to the
JSONs. No result of that run is in this README yet.

---

## Results

The generated GPU blocks show the stored JSON values. The hand-written
tables below them use three significant figures; the JSON and CSV files
keep full precision. Each figure is classified in
[`benchmarks/results/PROVENANCE.md`](benchmarks/results/PROVENANCE.md).
Commands and file names are in
[`benchmarks/results/README.md`](benchmarks/results/README.md).

### Measured — GPU, real prompts (MT-Bench and HumanEval, chat template)

This block is generated by `scripts/ingest_gpu_bundle.py --readme-commit 8d036d7`
from the result JSONs of the 0.2.0 bundle only (`benchmarks/results/gpu/8d036d7/`,
run from `main`). Do not edit it by hand. The earlier feature-branch
bundles (`36134c0`, the Phase 4 eager run that also covered TinyLlama, and
`52993e4`, the first compiled run) stay under `benchmarks/results/gpu/` as
history and are not shown here.

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

### Measured, 0.1.5 engine — TinyLlama-1.1B-Chat float32 · CPU · γ=4 · git `f59b5b4`

One HumanEval prompt, 16 new tokens, one drafter. Speedup is below 1.

| | FlashSpec UCB | Vanilla AR |
|---|---:|---:|
| tok/s | 1.75 | 4.30 |
| latency (s) | 9.14 | 3.72 |
| acceptance | 0.114 | unmeasured |
| speedup vs AR | 0.407 | unmeasured |

### Measured, 0.1.5 engine — TinyLlama-1.1B-Chat float16 · Tesla T4 · γ=4 · git `cb02ca1`

Eight synthetic prompts, one drafter. Speedup is below 1.

| | FlashSpec UCB | Vanilla AR |
|---|---:|---:|
| tok/s | 17.6 ± 1.43 | 29.0 ± 0.757 |
| p50 latency (ms) | 1860 | 1090 |
| p95 latency (ms) | 2140 | 1170 |
| acceptance | 0.208 ± 0.00825 | unmeasured |
| speedup vs AR | 0.606 ± 0.0501 | unmeasured |
| gpu memory (GB) | 2.38 | 2.38 |

## Microbenchmarks and historical

These rows are a random network, a sampling kernel, or the historical NF4 prompt. They sit below the paired end-to-end runs. The kernel and bandit measurements are under [Experimental components](#experimental-components).

### CPU toy and residual — git `da8182b`

Random in-process network, and a PyTorch residual microbench. The toy
speedup is below 1. The bandit row is a synthetic sampler, not that network.

| | FlashSpec UCB | Vanilla AR |
|---|---:|---:|
| tok/s | 808 ± 86.4 | 2320 ± 2.51 |
| p50 latency (ms) | 10.6 | 3.44 |
| acceptance | 0.312 ± 0.0600 | unmeasured |
| speedup vs AR | 0.348 ± 0.0376 | unmeasured |

| Vocab | PyTorch residual (ms) | PyTorch verification (ms) |
|---:|---:|---:|
| 1024 | 0.0182 ± 0.000935 | 0.418 ± 0.0221 |
| 4096 | 0.0197 ± 0.000205 | 0.523 ± 0.0435 |

| Arm | Acceptance | tok/s |
|---|---:|---:|
| UCB on the switch | 0.556 | 5790 |
| Fixed A | 0.200 | 2570 |
| Fixed B | 0.200 | 3220 |
| Single-drafter control | 0.200 | 2880 |

An older notebook cell assigns 44.2 tok/s, acceptance 0.75, p50 22.1 ms; no timing log exists, so it is not used as a result.

### Gamma sweep — measured (toy logprobs, synthetic)

Acceptance rate and throughput across speculation lengths γ ∈ {1, 2, 4, 8, 16}.
Data in [`benchmarks/results/gamma_sweep.csv`](benchmarks/results/gamma_sweep.csv).

| γ | α (mean) | Throughput (tok/s) | Step (ms) |
|---|---|---|---|
| 1 | 0.720 | 1,992 | 0.50 |
| 2 | 0.680 | 4,623 | 0.43 |
| 4 | 0.690 | 9,746 | 0.41 |
| 8 | 0.748 | 17,748 | 0.45 |
| 16 | 0.714 | 35,659 | 0.45 |

> Throughput here is the sampling kernel on random log-probs. The CSV
> column `measurement_kind` is `sampling_kernel_only_random_logprobs`.
> It is not end-to-end inference and it is not an H100 result.

No H100 comparison exists, and Medusa and EAGLE are not implemented here, so no comparison with them exists.

---

## Experimental components

### Bandit (experimental)

The tables in this README come from one-drafter runs. The selector was
measured choosing among seven arms once, in the Phase 6 run (bundle
`3e38251`; values in `benchmarks/results/gpu/3e38251/`, reported by
`scripts/phase6_report.py`), where it did not beat the best fixed arm on
the mixed workload. The regret numbers below come from a synthetic K=3
simulation.

#### Bandit regret — measured, synthetic (T=10,000, K=3 arms)

In this simulation UCB1 and Thompson sampling stay inside the
O(√(KT log T)) bound over 10,000 rounds.
UCB1 cumulative regret at T=10,000: **100.2** (theory bound: ~526).
Thompson cumulative regret at T=10,000: **18.9**.

![Bandit Regret](paper/figures/bandit_regret.jpg)

The figure is `paper/figures/bandit_regret.jpg`. The saved output is
`notebooks/02_bandit_analysis.ipynb` cell 3 (0-based index, execution
count 4): UCB1 100.2, Thompson 18.9, Oracle 0.0. This is a synthetic
K=3 arm experiment, not a throughput gain on a language model.

### Experimental kernels (not on the critical path)

The engine does not call these kernels. A speculative step's time is the
drafter and target forwards (milliseconds); the accept/reject test and
the residual are microseconds and run in PyTorch on the device.

- **Triton `verify_tokens` / `gather_accepted`** (`flashspec.kernels`).
  `verify_tokens` tiles over `(batch × γ)` positions; each thread reads
  two scalars (`log p`, `log q` at the draft token index) and writes one
  bool and one int32, so SRAM use does not grow with the vocabulary. It
  matches the pure-PyTorch reference numerically. There is no GPU CI
  runner: `notebooks/colab_triton_tests.py` runs the `@pytest.mark.gpu`
  parity tests on Colab and writes `colab_triton_tests.log`.
- **Assembly residual** (`flashspec-asm-kernel`, the `[asm]` extra,
  Linux x86-64). A residual evaluation is microseconds against
  millisecond model forwards, so it does not change end-to-end
  throughput.

#### Triton verification kernel vs the PyTorch reference — measured (Tesla T4)

Saved output of `notebooks/03_kernel_profiling.ipynb` cell 3 (0-based
index, execution count 4), printed there as a Tesla T4 run. At batch 1
the Triton kernel is slower than the PyTorch reference.

| Shape | Reference (ms) | Triton (ms) | Speedup |
|---|---|---|---|
| B=1 γ=4 V=32k | 0.151 | 0.785 | 0.2× |
| B=8 γ=4 V=32k | 0.717 | 0.459 | 1.6× |
| B=1 γ=8 V=32k | 0.231 | 1.248 | 0.2× |
| B=32 γ=4 V=32k | 0.091 | 0.222 | 0.4× |

> The Triton kernel is slower than the pure-PyTorch reference at batch 1
> on this T4 measurement. A faster H100 result has not been measured.
> Do not treat the H100 roofline in `docs/kernels.md` as a profile.

#### Assembly residual (`[asm]` extra) — Linux, git `4e48c55`, then git `1b0cda3`

Milliseconds, backend `asm`. Compare backends inside one run. The two runs do not repeat the same ordering.

| Vocab | PyTorch residual | C baseline | ASM residual | PyTorch verify | ASM verify |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0118 ± 0.00599 | 0.0109 ± 0.00177 | 0.0151 ± 0.00971 | 0.366 ± 0.0209 | 0.422 ± 0.0438 |
| 4096 | 0.0116 ± 0.000285 | 0.0140 ± 0.00214 | 0.0140 ± 0.00430 | 0.495 ± 0.0218 | 0.537 ± 0.0159 |
| 32000 | 0.0302 ± 0.00818 | 0.0430 ± 0.00268 | 0.0248 ± 0.0230 | 1.28 ± 0.638 | 1.32 ± 0.658 |

| Vocab | PyTorch residual | C baseline | ASM residual | PyTorch verify | ASM verify |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.00686 ± 0.00312 | 0.00609 ± 0.000588 | 0.00806 ± 0.00108 | 0.158 ± 0.0421 | 0.164 ± 0.0305 |
| 4096 | 0.00685 ± 0.000109 | 0.00785 ± 0.000559 | 0.00872 ± 0.000941 | 0.231 ± 0.0339 | 0.250 ± 0.0390 |
| 32000 | 0.0124 ± 0.00209 | 0.0235 ± 0.00266 | 0.0151 ± 0.0187 | 0.747 ± 0.393 | 0.730 ± 0.376 |

---

## How it works

Batch-1 fast path (`variant` `"rejection"` or `"greedy"`):

```mermaid
sequenceDiagram
    participant E as SpeculativeEngine
    participant B as Bandit (experimental)
    participant D as Drafter
    participant T as Target

    loop every decoding step
        E->>B: select() → arm index
        E->>D: draft γ tokens (γ forwards on its KV cache)
        E->>T: verify pending + γ tokens [one forward]
        E->>E: rejection_step or greedy_step on the device (PyTorch)
        E->>E: one host read: n_accepted, next token
        E->>D: commit: crop the cache, set the pending token
        E->>T: commit: crop the cache, set the pending token
        E->>B: update(arm, emitted tokens / step seconds)
    end
```

**Batching.** A batch of more than one row uses the generic loop. Rows that accept different numbers of tokens finish the step at different context lengths. Each loop forwards only the largest group of rows that still share a length. Other lengths wait for a later loop. The KV cache is rolled only when that group has a single row. A larger group full-forwards. The returned tensor fills a shorter row with `-1` so the tensor is rectangular. That fill is not an attention mask. The engine does not pad unequal prompts into one masked batch.

**Correctness.** The accept/reject criterion is `u_i < min(1, p(xᵢ)/q(xᵢ))`
computed in log-space to avoid underflow. Rejected positions sample a residual
token from `max(0, p−q) / ‖max(0, p−q)‖₁` (Algorithm 1, Leviathan et al. 2023).
Together these preserve the target distribution exactly.

**Compiled path.** With `FlashSpecConfig.compile` on a CUDA device, the
drafter and target run on preallocated static KV caches; a crop moves the
write position instead of slicing tensors, and the fixed-shape decode and
verify calls replay as CUDA graphs. The prompt prefill stays eager.

**Bandit (experimental).** UCB1 score for arm k at round t:
`μ̂ₖ + c · √(2 log t / nₖ)`. Unpulled arms are selected first. The reward
is emitted tokens per step second. In a synthetic K=3 simulation UCB1
regret at T=10,000 is 100.2 (theory bound ≈ 526) and Thompson's is 18.9.
On a language model it was measured once, over seven arms (Phase 6,
bundle `3e38251`), and lost to the best fixed arm on the mixed workload.

See [`docs/architecture.md`](docs/architecture.md) for the full component
diagram and proof sketch.

---

## Installation

```bash
# PyPI — cross-platform (Windows/macOS/Linux):
pip install flashspec

# PyPI — adds Triton (Linux only): the experimental Triton kernels.
# Linux CUDA builds of PyTorch already ship Triton, which torch.compile
# needs for the compiled path.
pip install flashspec[gpu]

# Optional AVX2 residual kernel. The wheel is a Linux x86-64 ELF library.
# On Windows and macOS the extra installs nothing and the PyTorch residual stays selected.
pip install flashspec[asm]

# Optional single-device HTTP API (FastAPI). One GPU or CPU, no streaming.
pip install flashspec[serve]

# From source:
git clone https://github.com/Mattral/FlashSpec && cd FlashSpec
pip install -e ".[dev,gpu]"   # omit ",gpu" on Windows/macOS

# Docker (Linux + CUDA). No image is published; build it locally:
docker build -f deploy/Dockerfile -t flashspec .
docker run --gpus all flashspec make test
```

> **Windows / macOS:** `pip install flashspec` works out of the box.
> Triton has no official PyPI wheels outside Linux, so the compiled path
> and the experimental Triton kernels are Linux + CUDA only. The engine's
> accept/reject test is PyTorch and runs everywhere PyTorch does. The
> kernels' pure-PyTorch reference is `flashspec.kernels._reference`.
> Check availability at runtime: `from flashspec import TRITON_AVAILABLE`.

### Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | |
| PyTorch | ≥ 2.2 | CPU or CUDA |
| Triton | ≥ 3.0 | Optional · Linux + CUDA · needed by the compiled path (Linux CUDA PyTorch ships it) and the experimental kernels |
| flashspec-asm-kernel | ≥ 0.1.2, < 2 | Optional · Linux x86-64 · `pip install flashspec[asm]`. 0.1.2 ships a manylinux wheel with `libflashspec_asm.so`. |
| FastAPI, uvicorn, httpx | see `serve` extra | Optional · `pip install flashspec[serve]` |
| CUDA | ≥ 12.0 | GPU path only |
| transformers | ≥ 4.40 | For HuggingFace model loading |

The `asm` extra installs `flashspec-asm-kernel` >= 0.1.2 and < 2 on Linux
x86-64. Release 0.1.2 is a manylinux wheel that contains
`libflashspec_asm.so`. Release 0.1.0 was an sdist with no shared library.
An import error, or an `OSError` from `abi_version()`, keeps the
pure-PyTorch residual. A residual evaluation is microseconds, and a model
forward is milliseconds, so this kernel does not change end-to-end
throughput. `benchmarks/results/residual_linux_gha.json` and
`benchmarks/results/residual_linux_gha_1b0cda3.json` are source builds
from the flashspec-asm repo, not a timing of the 0.1.2 wheel.
The PyPI project `flashspec` at version 0.1.2 was yanked by the maintainer.
Do not install `flashspec==0.1.2`. That yank does not apply to
`flashspec-asm-kernel`.

---

## Testing

```bash
make test           # unit + integration on CPU (no GPU required, ~2 min)
make test-chaos     # adversarial bandit tests
make test-gpu       # pytest -m gpu; needs CUDA. This repository has no GPU runner.
make bench-quick    # sampling-kernel smoke test, no model weights, writes no result file
make bench-cpu      # residual microbench, synthetic bandit demo, tiny_cpu.yaml
make bench          # runs every config in benchmarks/configs; skips any checkpoint this machine cannot load
```

The distribution check in the suite is a chi-squared test
(`tests/integration/test_output_distribution.py`, N = 30,000, vocabulary 6,
critical value 15.086 at 5 degrees of freedom and α = 0.01). It is not a
Kolmogorov–Smirnov test. The CPU test job runs it. Pytest fails under 83%
line coverage, the measured unit-and-integration coverage rounded down.
The CI lint job also runs `mypy --strict flashspec`.

---

## Future work

Not scheduled and not implemented:

- **A contextual bandit, run once.** The Phase 6 bandit had no context, and
  the best fixed arm changed between consecutive prompts of the mixed
  workload more often than not. A pre-registered design: the context is the
  next lookup's match length bucket ({0, 1-2, 3-4, >= 5}), which the step's
  single device read already carries; with no match the arms are Qwen at
  gamma 2, 4 and 6, otherwise n-gram and Qwen at gamma 2, 4 and 6; the value
  per (context, arm) is total emitted tokens / total seconds, not a mean of
  per-step ratios; UCB1, normalised, verify width 7. Its exit criteria would
  be fixed before the run (beat `composite-g4` and every fixed arm on the
  mixed workload in both modes, within 5% of each set's best fixed arm,
  faster than compiled HuggingFace everywhere, no new greedy mismatches),
  and a miss would be recorded without tuning or a second attempt. The
  oracle ceiling estimated from the Phase 6 bundle is small, so near parity
  with the composite strategy is the expected outcome.
- A larger target (Qwen2.5-3B-Instruct with the 0.5B drafter on the T4),
  the batch > 1 path, and draft log-probs kept in the model dtype.

---

## Repository layout

```
flashspec/          Installable package (engine, kernels, bandit, sampling, metrics)
benchmarks/         Runners, configs, and results/ for all paper numbers
tests/              Unit · integration · chaos suites (83% coverage floor)
notebooks/          Quickstart · bandit regret analysis · kernel profiling
docs/               MkDocs site (architecture, kernels, bandit, benchmarks)
paper/              Draft Zenodo preprint (article class) and bibliography
paper/joss/         JOSS-format draft, not under review
paper/figures/      Vector and raster figures (bandit_regret.jpg generated)
scripts/            CLI tools (export draft, download models, profile kernels)
deploy/             Dockerfile · docker-compose.yml · k8s manifests
```

---

## Links

| Resource | URL |
|---|---|
| Documentation | [flashspec.readthedocs.io](https://flashspec.readthedocs.io) |
| Preprint (Zenodo) | [10.5281/zenodo.20766119](https://doi.org/10.5281/zenodo.20766119) |
| Draft preprint | [`paper/flashspec.tex`](paper/flashspec.tex) |
| JOSS draft | [`paper/joss/paper.md`](paper/joss/paper.md) |
| Benchmark guide | [`benchmarks/README.md`](benchmarks/README.md) |
| Changelog | [`CHANGELOG.md`](CHANGELOG.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Publishing roadmap | [`PUBLISHING.md`](PUBLISHING.md) |
| Postponed work | [`docs/postponed.md`](docs/postponed.md) |

---

## Citation

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

**Preprint DOI**: [10.5281/zenodo.20766119](https://doi.org/10.5281/zenodo.20766119) (June 2026). That record is the citable archive. [`paper/flashspec.tex`](paper/flashspec.tex) is a later draft (October 2026, software 0.2.0) and has not been deposited.

Machine-readable metadata in [`CITATION.cff`](CITATION.cff).

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
