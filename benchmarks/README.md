# FlashSpec Benchmarks

How to reproduce every number in the paper.

---

## Quick smoke-test (no GPU or weights required)

```bash
make bench-quick
# or
python benchmarks/run_all.py --config benchmarks/configs/ --toy
python benchmarks/compare_baselines.py --toy
python benchmarks/sweep_gamma.py --toy
python benchmarks/sweep_draft_sizes.py --toy
```

---

## Full benchmark (requires H100 SXM5 + model weights)

```bash
# Pull model weights via HuggingFace Hub (requires HF_TOKEN env var).
python scripts/download_models.py

# Run all benchmarks — results written to benchmarks/results/.
make bench
```

---

## Reproducing individual paper tables and figures

### Table 1 — Main throughput comparison (Llama-3-8B, MT-Bench, γ=4)

```bash
python benchmarks/compare_baselines.py \
    --config benchmarks/configs/llama3_8b.yaml
```

Result file: `benchmarks/results/llama3_8b_<commit>.json`

### Table 2 — Throughput across models and datasets

```bash
python benchmarks/run_all.py --config benchmarks/configs/
```

### Figure 1 — γ sweep (acceptance rate vs speculation length)

```bash
python benchmarks/sweep_gamma.py
```

Result file: `benchmarks/results/gamma_sweep.csv`

### Figure 2 — Draft model size sweep (acceptance rate vs parameter count)

```bash
python benchmarks/sweep_draft_sizes.py
```

Result file: `benchmarks/results/draft_size_sweep.csv`

### Figure 3 — Bandit regret convergence

Run `notebooks/02_bandit_analysis.ipynb`.

---

## Benchmark result schema (AGENTS.md §14)

Every JSON result in `benchmarks/results/` contains:

```json
{
  "schema_version": "1.0",
  "timestamp": "2025-11-01T14:32:00Z",
  "git_commit": "abc1234",
  "hardware": {
    "gpu": "NVIDIA H100 SXM5 80GB",
    "driver": "550.54.15",
    "cuda": "12.4"
  },
  "model": "meta-llama/Llama-3-8B-Instruct",
  "method": "flashspec_ucb",
  "gamma": 4,
  "dataset": "mt_bench",
  "n_samples": 200,
  "metrics": {
    "tokens_per_second_mean": 142.3,
    "tokens_per_second_std": 4.1,
    "speedup_vs_ar_mean": 2.31,
    "speedup_vs_ar_std": 0.08,
    "acceptance_rate_mean": 0.73,
    "acceptance_rate_std": 0.05,
    "p50_latency_ms": 38.1,
    "p95_latency_ms": 44.2,
    "p99_latency_ms": 47.8,
    "gpu_memory_gb": 18.4
  }
}
```

---

## Adding a new benchmark

1. Create a new YAML config in `benchmarks/configs/`.
   Do **not** modify existing configs after their first committed result.
2. Add a row to the paper's results table referencing the new config.
3. Run `make bench` and commit the JSON output in `benchmarks/results/`.
4. Update this README with the reproduction command.

---

## Hardware and software versions

| Item | Value |
|------|-------|
| GPU | NVIDIA H100 SXM5 80 GB |
| CUDA | 12.4 |
| PyTorch | 2.3.0 |
| Triton | 3.0.0 |
| Transformers | 4.40.0 |

Exact versions are recorded in every result JSON file.
