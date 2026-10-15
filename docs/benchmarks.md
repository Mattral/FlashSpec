# FlashSpec Benchmarks

## Methodology

All benchmarks follow the **immutable result schema** defined in AGENTS.md §14.
Once a result JSON is committed, the config that produced it is frozen.

### Hardware the paper configs were written for

These rows describe the machine the Llama configs target. They are not
a record of a run. This checkout has no committed H100 result. A config
whose device is missing, or whose checkpoint cannot be loaded, writes
no JSON.

| Field        | Intended paper machine |
|--------------|------------------------|
| GPU          | NVIDIA H100 SXM5       |
| CUDA         | 12.4                   |
| Driver       | 550.x                  |
| Precision    | bfloat16               |
| Batch size   | 1                      |

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
| `medusa`          | Not implemented. The harness skips it and writes nothing. |
| `eagle`           | Not implemented. The harness skips it and writes nothing. |
| `flashspec_ucb`   | FlashSpec with UCB1 bandit                |
| `flashspec_thompson` | FlashSpec with Thompson sampling        |

---

## Running benchmarks

```bash
# Fast smoke-test (no GPU or weights required). Writes no quoted result.
make bench-quick

# This machine's CPU measurements: residual, synthetic bandit, toy model.
make bench-cpu

# Full benchmark (requires GPU + model weights):
make bench
```

`make bench` and `make bench-cpu` write JSON under `benchmarks/results/`.
The Windows CPU files from this checkout are listed in the README
section "This checkout". `make bench-quick` is a smoke test.

`notebooks/04_gpu_measurement.ipynb` is the GPU measurement. A Tesla T4
is enough for the TinyLlama pair. Run it in Colab, download
`flashspec_bring_back.zip`, and copy the JSON into `benchmarks/results/`.
Google Drive is not this repository.

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
