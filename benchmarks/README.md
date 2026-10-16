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

`make bench-quick` is a sampling-kernel smoke test. It does not write a
result file you should quote.

`make bench-cpu` runs three commands and writes JSON under
`benchmarks/results/`:

```bash
python benchmarks/bench_residual.py
python scripts/demo_bandit_mixed.py
python benchmarks/run_all.py --config benchmarks/configs/tiny_cpu.yaml
```

On Windows, if `make` is not installed, run those three Python commands.
The files are a CPU residual microbench, a synthetic two-drafter switch,
and a randomly initialised transformer. They are not Llama and not H100.

---

## Full benchmark

`make bench` runs every YAML file in `benchmarks/configs/`. A config that
asks for CUDA when CUDA is absent, or names a checkpoint that cannot be
loaded, writes nothing and the process exits 2 after the configs that
could run. `benchmarks/configs/tiny_cpu.yaml` is an in-process random
network. Its JSON is a CPU toy measurement, not Llama and not H100.

Medusa and EAGLE are not implemented. The harness prints that and writes
no file for them.

## Fixed prompt sets

`benchmarks/prompt_sets/` holds four MIT HumanEval prompts and four
Apache-2.0 MT-Bench first-turn questions. The harness tokenizes them
with the target tokenizer and writes one JSON per dataset. The document
includes `prompt_ids` and the existing `acceptance_rate_mean` field.
Stanford Alpaca is CC BY-NC 4.0 and its text is not in the tree, so a
config that names it writes nothing for that name. One CPU float32
sample of HumanEval/23 is recorded at git `f59b5b4` in
`benchmarks/results/tinyllama_cpu_humaneval_flashspec_ucb_f59b5b4.json`
and `benchmarks/results/tinyllama_cpu_humaneval_vanilla_ar_f59b5b4.json`.
MT-Bench has no acceptance number in this tree. Attribution and the
license copies are in `benchmarks/prompt_sets/README.md`.

`benchmarks/configs/llama3_8b.yaml` already lists `mt_bench`,
`humaneval`, and `alpaca`. This section does not add a config file.
`make bench` still globs every YAML, and that Llama config asks for
CUDA and gated weights.

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

The recorded figure is `paper/figures/bandit_regret.jpg`.
`notebooks/02_bandit_analysis.ipynb` is in this checkout. The unit test
`test_ucb1_regret_within_theoretical_bound` also checks the bound.
Cumulative regret at T=10,000 was UCB1 100.2 and Thompson 18.9.

---

## Benchmark result schema (AGENTS.md §14)

`benchmarks/results/baseline.json` is a schema template. Its `status` is
`template` and its `metrics` object is empty. It is not a measurement.
The numbers in the example below are the shape of a measured file. They
are not a result from this repository. A measured file sets
`"status": "measured"`, omits metric keys it did not record, and lists
those keys under `unmeasured`. Copying the unmeasured design target 142.3 tok/s
figure into a JSON file is a failure of the validator when the hardware
string contains H100 and the file is not a real measured run.

```json
{
  "status": "measured",
  "measurement_kind": "end_to_end",
  "timestamp": "2026-01-01T00:00:00+00:00",
  "git_commit": "abc1234",
  "hardware": {"platform": "recorded by the harness"},
  "model": "the checkpoint that was loaded",
  "method": "flashspec_ucb",
  "gamma": 4,
  "dataset": "the prompt source",
  "n_samples": 200,
  "metrics": {"tokens_per_second_mean": 0.0},
  "unmeasured": [],
  "notes": "Replace this example. Do not commit it as a result."
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

Record them in the result JSON that the harness writes. Do not copy a
target-machine table into a file that was produced on different hardware.
