# Provenance

Labels used before a number is moved or removed:

- **MEASURED** — a committed JSON, CSV, or notebook output computed by the
  cell that saved it. Cite the file.
- **LOGGED** — a committed notebook output or write-up. Cite it. Say when
  the cell assigns the figure instead of timing it.
- **UNVERIFIED** — no surviving output. Keep the figure only in a labelled
  historical section. Do not delete it silently.
- **TARGET** — a goal. It was not measured. Remove it from a comparison
  with methods this repository does not implement.

Do not describe a target or an unverified figure as invented or fabricated.
Commit `78add0d` used "invented" for the H100 design-target table. Later
text says "unmeasured design target".

`~1800 ms saved per hit` has no committed run behind it. That phrase is
not in this tree, and it is not added here.

## MEASURED

| Figure | Evidence |
|---|---|
| CPU float32 TinyLlama, HumanEval/23, git `f59b5b4`: FlashSpec 1.7507264092133707 tok/s, vanilla autoregressive 4.2977579080394 tok/s, speedup 0.40735807988124606, acceptance 0.11363636363636363 | `benchmarks/results/tinyllama_cpu_humaneval_flashspec_ucb_f59b5b4.json`, `benchmarks/results/tinyllama_cpu_humaneval_vanilla_ar_f59b5b4.json` |
| Colab Tesla T4 float16, git `cb02ca1`: FlashSpec 17.576979879136424 tok/s, vanilla 29.006488905704696 tok/s, speedup 0.6061833262227975, acceptance 0.20834964273981926, gpu memory 2.3778347969055176 GB | `benchmarks/results/tinyllama_gpu_flashspec_ucb_cb02ca1.json`, `benchmarks/results/tinyllama_gpu_vanilla_ar_cb02ca1.json`, chart `benchmarks/results/tinyllama_throughput.png` |
| CPU toy, git `da8182b`: FlashSpec 807.876431892803 tok/s, vanilla 2322.725711556072 tok/s, speedup 0.3478541784757594 | `benchmarks/results/tiny_cpu_flashspec_ucb_da8182b.json`, `benchmarks/results/tiny_cpu_vanilla_ar_da8182b.json` |
| CPU residual microbench | `benchmarks/results/residual_cpu.json` |
| Linux ASM residual, git `4e48c55` and git `1b0cda3` | `benchmarks/results/residual_linux_gha.json`, `benchmarks/results/residual_linux_gha_1b0cda3.json` |
| Synthetic bandit switch | `benchmarks/results/bandit_mixed_workload.json` |
| Gamma sweep, sampling kernel on random log-probs: α 0.720, 0.680, 0.690, 0.748, 0.714 and the tok/s column | `benchmarks/results/gamma_sweep.csv` |
| Draft-size sweep | `benchmarks/results/draft_size_sweep.csv` |
| Tesla T4 kernel table, including 0.151 ms reference and 0.785 ms Triton at B=1, γ=4, V=32k (0.2×) | `notebooks/03_kernel_profiling.ipynb` cell 3 (0-based index, execution count 4). The cell prints the table from timed calls. |
| Synthetic K=3 regret: UCB1 100.2, Thompson 18.9, Oracle 0.0, T=10,000 | `notebooks/02_bandit_analysis.ipynb` cell 3 (0-based index, execution count 4). The cell prints the mean of the simulation. |

README tables round these measured files to three significant figures.
The files keep full precision.

## LOGGED

| Figure | Evidence |
|---|---|
| NF4 TinyLlama on Tesla T4: 44.2 tok/s, acceptance 0.75, p50 22.1 ms | `notebooks/Colab_Benchmarks_T4.ipynb` cell 25 (0-based index, execution count 63) and `benchmarks/results/flashspec_ucb_tiny_llama.json`. The saved output shows the checkpoint load, then the three fields. The cell source assigns those three fields after the load. The maintainer recalls a timing log from an earlier machine. That log is not in this repository. Speedup, p95, p99, and GPU memory are 0.0 in the JSON. No vanilla autoregressive baseline was recorded. |

The 44.2 row stays logged. The recollection does not add a file to this
tree and does not change the classification.

## UNVERIFIED

| Figure | Evidence |
|---|---|
| `~1800 ms saved per hit` | No committed run. The phrase is not in this tree. |

## TARGET

| Figure | Evidence |
|---|---|
| Llama-3-8B H100 table: vanilla 61.4 / 61.1, Medusa 98.7 / 95.2, EAGLE 112.3 / 109.8, FlashSpec 142.3 / 138.9, and the paired acceptance and speedup columns | Unmeasured design targets. They were in the README until `78add0d` and in `paper/flashspec.tex` until the table was removed. Medusa and EAGLE are not implemented, so the comparison is not shown. The numbers were not found in a result JSON. |
| `docs/benchmarks.md` speedup ≥ 2.0× and acceptance ≥ 0.65 | Targets for a Llama-3-8B run. No such run is committed. |
| `docs/kernels.md` H100 roofline arithmetic | Arithmetic from the NVIDIA H100 whitepaper. Not a profile of a run in this repository. |
| `benchmarks/README.md` warning that names 142.3 tok/s | A reminder not to copy an unmeasured design target into a result file. |

## Asking before a downgrade

A MEASURED or LOGGED row stays until the maintainer says the underlying
log does not exist. For the NF4 row, the maintainer recalls a timing log
from an earlier machine. That log is not in this repository. The row
stays LOGGED. The notebook cell still assigns the three fields after the
checkpoint load.
