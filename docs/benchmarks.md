# FlashSpec Benchmarks

## Methodology

Committed result JSON follows the schema described in
`benchmarks/README.md`. `benchmarks/results/baseline.json` is a template,
not a measurement. Once a result JSON is committed, the config that
produced it stays as it was committed. `AGENTS.md` is not in this checkout.

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

The Llama configs under `benchmarks/configs/` set 50 warmup steps. The
committed CPU HumanEval/23 pair at git `f59b5b4` is recorded in
`benchmarks/results/README.md` with `n_warmup` 0. That field is not in
the result JSON itself. Cite the notes for the run you quote.

### Datasets

- **MT-Bench**: conversational multi-turn benchmark. No MT-Bench generation
  result is committed.
- **HumanEval**: Python code generation. The committed CPU pair uses
  HumanEval/23.
- **Alpaca**: named as an instruction-following set. The Alpaca text is
  not in this tree, and no Alpaca generation result is committed.

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
CPU result files live under `benchmarks/results/`. The root README
does not name the host machine. `make bench-quick` is a smoke test.

`notebooks/04_gpu_measurement.ipynb` is the GPU measurement. The Colab
Tesla T4 run at git `cb02ca1` is in
`benchmarks/results/tinyllama_gpu_flashspec_ucb_cb02ca1.json` and
`benchmarks/results/tinyllama_gpu_vanilla_ar_cb02ca1.json`. A later GPU
run still has to be downloaded from Colab and copied here. Google Drive
is not this repository. A CPU float32 HumanEval/23 pair at git `f59b5b4`
is in `benchmarks/results/tinyllama_cpu_humaneval_flashspec_ucb_f59b5b4.json`
and `benchmarks/results/tinyllama_cpu_humaneval_vanilla_ar_f59b5b4.json`.

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
