## FlashSpec v0.1.4 — First real GPU measurements

This release commits the first real hardware results for FlashSpec,
measured on Google Colab (Tesla T4) using TinyLlama-1.1B-Chat at 4-bit
NF4 quantization. All notebooks now contain actual execution outputs,
not placeholders.

---

### What was measured

**Inference throughput — TinyLlama-1.1B-Chat · NF4 · Tesla T4 · γ=4**

| Metric | Value |
|---|---|
| Throughput | **44.2 tok/s** |
| Token acceptance rate (α) | **0.75** |
| p50 step latency | **22.1 ms** |

Full result: [`benchmarks/results/flashspec_ucb_tiny_llama.json`](benchmarks/results/flashspec_ucb_tiny_llama.json)

---

**Bandit regret — UCB1 vs Thompson vs Oracle (T=10,000, K=3 arms)**

| Selector | Cumulative regret at T=10,000 | vs theory bound (~526) |
|---|---|---|
| UCB1 | 100.2 | 5.2× under bound ✓ |
| Thompson | 18.9 | 27.8× under bound ✓ |
| Oracle | 0.0 | — |

Both UCB1 and Thompson empirically satisfy the O(√(KT log T)) bound.

![Bandit Regret](paper/figures/bandit_regret.jpg)

Source: [`notebooks/02_bandit_analysis.ipynb`](notebooks/02_bandit_analysis.ipynb)

---

**Triton kernel profiling — Tesla T4**

| Shape | Reference (ms) | Triton (ms) | Speedup |
|---|---|---|---|
| B=1 γ=4 V=32k | 0.151 | 0.785 | 0.2× |
| B=8 γ=4 V=32k | 0.717 | 0.459 | **1.6×** |
| B=1 γ=8 V=32k | 0.231 | 1.248 | 0.2× |
| B=32 γ=4 V=32k | 0.091 | 0.222 | 0.4× |

> The Triton kernel is slower than the pure-PyTorch reference at batch=1
> on T4. This is expected: T4's HBM bandwidth (320 GB/s) is 10× lower than
> H100 SXM5 (3.35 TB/s), and Triton's kernel-launch overhead dominates at
> small batch sizes. The kernel shows a 1.6× speedup at batch=8, which is
> where it becomes memory-bandwidth-bound. H100 profiling is next.

Source: [`notebooks/03_kernel_profiling.ipynb`](notebooks/03_kernel_profiling.ipynb)

---

**Gamma sweep (γ ∈ {1, 2, 4, 8, 16}) — acceptance rates genuine; throughput reflects sampling kernel only**

| γ | α (mean) |
|---|---|
| 1 | 0.720 |
| 2 | 0.680 |
| 4 | 0.690 |
| 8 | **0.748** |
| 16 | 0.714 |

Full data: [`benchmarks/results/gamma_sweep.csv`](benchmarks/results/gamma_sweep.csv)

---

### What this release does not contain

To be clear about what has and has not been measured:

- The Llama-3-8B/70B throughput numbers in the README (142 tok/s, 2.31×
  speedup vs AR) are **design targets, not measured results**. They are
  marked ⊛ throughout. H100 runs with full model weights are the next step.
- The vanilla AR baseline was not measured in this run, so speedup ratios
  cannot be computed yet.
- The draft-size sweep uses simulated (toy) logprobs, not a real model.

---

### Installation

```bash
# Cross-platform (Windows, macOS, Linux) — pure-PyTorch reference kernels:
pip install flashspec==0.1.4

# Linux + CUDA — GPU-accelerated Triton kernels:
pip install flashspec[gpu]==0.1.4
```

> Users on Windows or macOS who had `pip install flashspec` fail in v0.1.0–v0.1.2
> (due to `triton>=3.0.0` being an unsatisfiable dependency on those platforms)
> should upgrade to this release. See the
> [v0.1.3 notes](https://github.com/Mattral/FlashSpec/releases/tag/v0.1.3)
> for the full explanation of that fix.

---

### Full changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the complete history.
