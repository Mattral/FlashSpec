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
- `tinyllama_gpu_*.json` and `larger_gpu_*.json`, after
  `notebooks/04_gpu_measurement.ipynb` has been run and the downloaded
  files have been copied here
- `residual_colab_cpu.json`, only if that notebook's ASM cell wrote it

Add another negation in `.gitignore` only for a file you have actually produced.
