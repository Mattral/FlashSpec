# FlashSpec Benchmarks

## Methodology

All benchmarks follow the **immutable result schema** defined in AGENTS.md §14.
Once a result JSON is committed, the config that produced it is frozen.

### Hardware

| Field        | Value              |
|--------------|--------------------|
| GPU          | NVIDIA H100 SXM5   |
| CUDA         | 12.4               |
| Driver       | 550.x              |
| Precision    | bfloat16           |
| Batch size   | 1                  |

### Warmup

50 warmup steps are discarded before measurement begins.

### Datasets

- **MT-Bench**: conversational multi-turn benchmark.
- **HumanEval**: Python code generation.
- **Alpaca**: instruction-following (subset 500).

---

## Baselines

| Method            | Description                               |
|-------------------|-------------------------------------------|
| `vanilla_ar`      | Standard autoregressive generation        |
| `medusa`          | Cai et al. (2024), multi-head draft heads |
| `eagle`           | Li et al. (2024), feature-level drafting  |
| `flashspec_ucb`   | FlashSpec with UCB1 bandit                |
| `flashspec_thompson` | FlashSpec with Thompson sampling        |

---

## Running benchmarks

```bash
# Fast smoke-test (no GPU or weights required):
make bench-quick

# Full benchmark (requires GPU + model weights):
make bench
```

Results are written to `benchmarks/results/` as JSON files.

---

## Performance targets (Llama-3-8B-Instruct, γ=4)

| Metric                         | Target   |
|-------------------------------|----------|
| Speedup vs vanilla AR         | ≥ 2.0×   |
| Regression threshold          | < 10% vs prior commit |
| Acceptance rate α             | ≥ 0.65   |

---

## Adding a new benchmark

1. Create a YAML config in `benchmarks/configs/`.
2. Do not modify existing configs after first commit.
3. Add the config name to the benchmark workflow in `.github/workflows/benchmark.yml`.
