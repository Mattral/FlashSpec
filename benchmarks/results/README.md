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

Add another negation in `.gitignore` only for a file you have actually produced.
