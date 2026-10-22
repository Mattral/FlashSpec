# Benchmark results

`baseline.json` is a schema template. `status` is `template` and `metrics`
is empty. It is not a measurement.

`gamma_sweep.csv` has `measurement_kind=sampling_kernel_only_random_logprobs`.
The tok/s column is the rejection sampler on random log-probabilities.

`draft_size_sweep.csv` has
`measurement_kind=toy_hidden_dim_proxy_not_a_real_model`. The size labels
are hidden-dimension proxies, not parameter counts of real draft models.

JSON files written by `benchmarks/harness.py`, `benchmarks/bench_residual.py`,
and `scripts/demo_bandit_mixed.py` are measurements of the machine that
ran them. Read `hardware`, `measurement_kind`, and `notes` before quoting
a number. A file whose hardware string contains H100 is rejected by
`scripts/update_benchmark_results.py` unless `status` is `measured` and
the metrics are not the all-zero placeholder.

`flashspec_ucb_tiny_llama.json` is the 0.1.3 single-prompt T4 record.
Fields stored as `0.0` in that file were not measured. Do not invent a
replacement. A new paired run comes from
`notebooks/04_gpu_measurement.ipynb`.

Most `*.json` files here are gitignored. These files are exceptions
because this checkout produced them:

- `baseline.json` (schema template, not a measurement)
- `residual_cpu.json`
- `bandit_mixed_workload.json`
- `tiny_cpu_*.json`
- `tinyllama_gpu_flashspec_ucb_cb02ca1.json` and
  `tinyllama_gpu_vanilla_ar_cb02ca1.json` are the Colab Tesla T4
  float16 run. `tinyllama_throughput.png` is the bar chart of those
  two means. `larger_gpu_*.json` is still absent.
- `tinyllama_cpu_humaneval_flashspec_ucb_f59b5b4.json` and
  `tinyllama_cpu_humaneval_vanilla_ar_f59b5b4.json` are the CPU float32
  run at git `f59b5b4`: one HumanEval/23 prompt, 16 new tokens, no
  warmup, one thread. FlashSpec was slower than vanilla autoregressive
  decoding on that sample.
- `residual_linux_gha.json` is the GitHub-hosted Linux run at git
  `4e48c55` where the ASM backend loaded.
  `residual_linux_gha_1b0cda3.json` is the second run, at git
  `1b0cda3`. In that file the ASM residual mean is higher than PyTorch
  at vocab 1024, 4096, and 32000. `residual_colab_cpu.json` was not
  produced. The Colab ASM cell failed.

## Provenance

Root README tables use three significant figures. These files keep full
precision.

### CPU float32 TinyLlama, git `f59b5b4`

`tinyllama_cpu_humaneval_flashspec_ucb_f59b5b4.json` and
`tinyllama_cpu_humaneval_vanilla_ar_f59b5b4.json`.

Target `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, drafter
`Felladrin/Llama-160M-Chat-v1`, `dataset: humaneval`, prompt
HumanEval/23, `n_samples: 1`, `max_new_tokens: 16`, `n_warmup: 0`,
`prompt_length: 128`, `batch_size: 1`, `gamma: 4`, `temperature: 1.0`,
`seed: 0`, float32, one thread. The YAML was not added under
`benchmarks/configs/`. The command was
`python benchmarks/run_all.py --config <that file> --output-dir <dir>`,
with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` set to 1. `run_all.py` calls
`torch.set_num_threads(1)`. `n_samples` is 1, so the standard deviation
is 0 and the nearest-rank percentiles match. `gpu_memory_gb` is
unmeasured. Hardware fields are platform, machine, Python, torch, and
AVX2. Acceptance was 0.11363636363636363 with one drafter, so the bandit
could not adapt. The vanilla file records HuggingFace `generate()` with
the KV cache. FlashSpec mean was 1.7507264092133707 tok/s and vanilla
autoregressive decoding was 4.2977579080394 tok/s
(`speedup_vs_ar_mean` 0.40735807988124606).

### CPU toy, git `da8182b`

`tiny_cpu_flashspec_ucb_da8182b.json` and
`tiny_cpu_vanilla_ar_da8182b.json` are
`measurement_kind=toy_random_init_transformer`, `n_samples=3`, γ=4,
synthetic prompts, one drafter, threads pinned to 1. `gpu_memory_gb` is
unmeasured. Speculative decoding was slower than vanilla autoregressive
sampling (`speedup_vs_ar_mean` 0.3478541784757594).

`residual_cpu.json` is `cpu_residual_and_verification_microbench`, 30
iterations, γ=4, random log-probs. `flashspec_asm_kernel` did not load.
ASM and the C baseline are listed under `unmeasured`.

`bandit_mixed_workload.json` is one
`controlled_mixed_workload_simulation` (`n_samples=1`, vocab 32,
`max_new_tokens=24`, seed 0, `window_size=4`). The target peak flips
from token 3 to token 9 after context length 16. On that switch, UCB
acceptance was 0.5555555555555556 and throughput was 5794.162381923592
tok/s. Fixed drafters were 0.2 / 2574.002574098843 tok/s and 0.2 /
3215.4340838834096 tok/s. The single-drafter control matched fixed A
(0.2 / 2876.111497299474 tok/s). `improved_acceptance_vs_best_fixed`
and `improved_throughput_vs_best_fixed` are both true for this run.
`speedup_vs_vanilla_ar`, `llama_acceptance`, and `gpu_throughput` are
unmeasured. These tok/s figures are the synthetic sampler.

### Linux residual

`residual_linux_gha.json` is a GitHub-hosted Linux run, Python 3.11.17,
torch 2.14.1+cpu, git `4e48c55`, AVX2 true, 30 iterations, one thread,
backend `asm` / `selected`. The clock is `perf_counter` around the Python
call. At vocab 1024 and 4096 the ASM residual mean is higher than pure
PyTorch. At vocab 32000 the ASM residual mean is lower than both PyTorch
and the C baseline, and its standard deviation is almost as large as the
mean. Verification with the ASM backend selected is slower than
pure-PyTorch verification at all three sizes.

`residual_linux_gha_1b0cda3.json` is a second run on the same class of
GitHub-hosted machine, git `1b0cda3`, timestamp
`2026-10-15T07:18:40.745328+00:00`, same Python and torch, backend `asm`
/ `selected`, 30 iterations, one thread. In that file the ASM residual
mean is higher than pure PyTorch at all three sizes. At vocab 32000 the
ASM verification mean is lower than pure PyTorch, and the two standard
deviations overlap. Compare backends inside one file.

### Colab Tesla T4 float16, git `cb02ca1`

`tinyllama_gpu_flashspec_ucb_cb02ca1.json` and
`tinyllama_gpu_vanilla_ar_cb02ca1.json`. The chart is
`tinyllama_throughput.png`.

Colab Tesla T4, Linux, Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0,
2026-10-15. Target `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, drafter
`Felladrin/Llama-160M-Chat-v1` from `benchmarks/configs/tinyllama_gpu.yaml`
at that commit. Weights are float16. `n_samples=8`, 32 new tokens,
prompt length 16, dataset `synthetic_token_ids`. One drafter is loaded.
Percentiles are nearest-rank, so p95 and p99 are the same sample.
`gpu_memory_gb` is the process peak after both models were resident
(2.3778347969055176 in both files). FlashSpec mean was
17.576979879136424 tok/s and vanilla autoregressive decoding was
29.006488905704696 tok/s (`speedup_vs_ar_mean` 0.6061833262227975).
The optional ASM cell printed `Residual bench failed` and did not write
`residual_colab_cpu.json`. That cell had installed PyPI
`flashspec-asm-kernel` 0.1.0. The larger-model cell was skipped.

### Older notebook cell

An older notebook cell assigns 44.2 tok/s, acceptance 0.75, p50 22.1 ms; no timing log exists, so it is not used as a result. The file `flashspec_ucb_tiny_llama.json` is kept. Its `_notes.provenance` field says the values were assigned in notebook cell 25 and were not timed.

### Other recorded rows

`gamma_sweep.csv` is the sampling kernel on random log-probs
(`measurement_kind=sampling_kernel_only_random_logprobs`).
`paper/figures/bandit_regret.jpg` and the K=3 regret numbers (UCB1 100.2,
Thompson 18.9, T=10,000) are a synthetic arm experiment.
`notebooks/03_kernel_profiling.ipynb` is the historical Tesla T4 kernel
profile. At batch 1 the Triton kernel was slower than the PyTorch
reference (0.2× at B=1, γ=4, V=32k).

Add another negation in `.gitignore` only for a file you have actually produced.
