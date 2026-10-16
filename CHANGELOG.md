# Changelog

All notable changes to FlashSpec are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/).

---

## [Unreleased]

Postponed work is listed in [`docs/postponed.md`](docs/postponed.md).
Those items are not blockers. v0.2.0 stays reserved for a measured
datacenter-GPU Llama run. Do not tag 0.2.0 before that measurement
exists.

### Changed
- The `asm` extra requires `flashspec-asm-kernel` >= 0.1.2 and < 2 on
  Linux x86-64 (`sys_platform == "linux"`). The kernel runs only when
  each probability row and the residual output row are CPU float32,
  contiguous, and do not require grad. A missing library, including the
  0.1.0 `OSError` or `ImportError`, keeps the pure-PyTorch residual.
  Release 0.1.2 ships `libflashspec_asm.so` in a manylinux wheel.

### Added
- `notebooks/04_gpu_measurement.ipynb` builds flashspec-asm from source
  when the installed package does not select the ASM backend.
- CPU tests for typical acceptance, the Linux AVX2 flag reader, ONNX
  export guards, and the serve command wiring.

---

## [0.1.5] — 2026-10-15 — Honest CPU path, optional ASM, small server

Development status stays Alpha. This version does not claim a production
deployment or an H100 speedup.

### Added
- Optional extra `asm`: `flashspec-asm-kernel` on Linux x86-64 only.
  The CPU residual uses that kernel when the ABI is 1, AVX2 is present,
  and the inputs are contiguous float32. Every other case, including a
  kernel exception, stays on pure PyTorch. AVX2 `vmaxps` maps NaN to 0;
  PyTorch preserves NaN. Finite inputs are required to match within 1e-5.
  Upstream Linux timings live in flashspec-asm `bench/results/phase5_benchmark.txt`
  and are not copied into this repo.
- Optional extra `serve`: one worker, a queue, batches only when prompt
  length and `max_tokens` match, and a hand-rolled Prometheus text endpoint.
  `POST /v1/chat/completions` and `POST /v1/completions`. No streaming,
  no continuous batching, no paged KV, no multi-GPU. Temperature and gamma
  are fixed at process start. The JSON `model` field does not switch
  checkpoints.
- `load_drafters` / `register_hf` for two or more HuggingFace drafters.
  A single arm logs that the bandit cannot adapt.
- `benchmarks/harness.py` writes a measured JSON or nothing. CUDA without
  a GPU, and a checkpoint that fails to load, write no file.
  Checkpoint loads use the config `dtype` (`float32`, `float16`, or
  `bfloat16`). A drafter and target with different vocabulary sizes
  raise before any result file is written.
  `benchmarks/configs/tiny_cpu.yaml` is an in-process random network.
  `benchmarks/configs/tinyllama_gpu.yaml` and
  `notebooks/04_gpu_measurement.ipynb` are the T4-or-newer measurement.
  The notebook downloads a zip; Colab Drive is not the repo.
- `benchmarks/bench_residual.py` and `scripts/demo_bandit_mixed.py`.

### Changed
- `SpeculativeEngine.generate` keeps one context per row. Rows that accept
  different lengths no longer have to be rectangular for `Tensor.view`.
  Tokens after the first rejection are discarded. A fully accepted step
  keeps the draft tokens. A residual whose mass is ~0 (draft matches
  target) samples from the target distribution instead of calling
  multinomial on an all-zero vector.
- `benchmarks/results/baseline.json` is a schema template with empty
  metrics. The previous all-zero H100 document was not a run.
- Gamma and draft-size CSVs carry a `measurement_kind` column.
- Version synced to 0.1.5 in `pyproject.toml`, `flashspec/__init__.py`,
  and `CITATION.cff`.

### Fixed
- PyPI `flashspec-asm-kernel` 0.1.0 is a source distribution with no
  `libflashspec_asm.so`. The package imports, then the first call raises
  `ImportError`. That error keeps the pure-PyTorch residual, so
  `import flashspec` succeeds on Linux after `pip install flashspec[asm]`.
- `benchmarks/bench_residual.py` writes `timestamp` and lists the
  end-to-end metric names in `unmeasured`. `residual_cpu.json` records
  the `cb02ca1` commit time `2026-10-15T06:26:21Z` for the timestamp
  the original script omitted. Its millisecond values are unchanged.

### Measured here
CPU toy run, git `da8182b`, one thread. The host machine is not named
in this changelog. The Linux ASM library did not load, so ASM and the
C baseline are unmeasured on that run. Full figures are in the JSON
files under `benchmarks/results/`.

- Toy random transformer, `n_samples=3`, γ=4, one drafter:
  FlashSpec 807.876431892803 tok/s, acceptance 0.3115079365079365,
  speedup vs vanilla AR 0.3478541784757594. Vanilla AR
  2322.725711556072 tok/s. Speculative decoding was slower on this toy.
- Residual microbench, 30 iterations, pure PyTorch: V=1024 residual
  0.018213333290380735 ms, verification 0.4179066666741467 ms; V=4096
  residual 0.01968333341816712 ms, verification 0.522750000012214 ms.
- Synthetic two-drafter switch, `n_samples=1`: UCB acceptance
  0.5555555555555556 and 5794.162381923592 tok/s, against fixed-drafter
  peaks of 0.2 acceptance and 3215.4340838834096 tok/s. That win is
  this controlled switch. It is not a language-model result, and those
  tok/s are not comparable to the toy end-to-end row.
- GitHub-hosted Linux runner, git `4e48c55`, torch 2.14.1+cpu, AVX2,
  30 iterations. `flashspec-asm` loaded (`backend=asm`). Residual means
  in ms: vocab 1024 PyTorch 0.011835766664110755, C 0.010876566682327393,
  ASM 0.01513639999378332; vocab 4096 PyTorch 0.011573599992213227,
  C 0.014012900006340109, ASM 0.014001300003959235; vocab 32000 PyTorch
  0.030213800005185476, C 0.04299199999877601, ASM 0.024774433317512983.
  Verification with ASM selected was slower than pure PyTorch at all
  three sizes. File: `benchmarks/results/residual_linux_gha.json`.
- Second GitHub-hosted Linux run, git `1b0cda3`, timestamp
  `2026-10-15T07:18:40.745328+00:00`, same Python and torch, backend
  `asm`. Residual means in ms: vocab 1024 PyTorch 0.0068616666718905135,
  C 0.006093900015002873, ASM 0.008056400001047829; vocab 4096 PyTorch
  0.006851300016326907, C 0.007845133351717474, ASM 0.008722400002625363;
  vocab 32000 PyTorch 0.012439799998749853, C 0.02345496665914955, ASM
  0.015084966666260394. ASM residual means are higher than PyTorch at
  all three sizes. The vocab-32000 ordering from the first file did not
  repeat. File: `benchmarks/results/residual_linux_gha_1b0cda3.json`.
- Colab Tesla T4, git `cb02ca1`, Python 3.13.15, torch 2.11.0+cu130,
  CUDA 13.0. TinyLlama-1.1B-Chat float16 with drafter
  `Felladrin/Llama-160M-Chat-v1`, `n_samples=8`, synthetic token ids,
  γ=4. FlashSpec 17.576979879136424 tok/s, acceptance
  0.20834964273981926, speedup vs vanilla AR 0.6061833262227975.
  Vanilla AR 29.006488905704696 tok/s. Speculative decoding was slower
  on this pair. Process peak memory 2.3778347969055176 GB. The ASM
  cell on that runtime failed and wrote no file.

---

## [0.1.4] — 2026-06-15 — First real GPU measurements

### Added
- First real GPU measurements (Tesla T4, TinyLlama-1.1B-Chat NF4, FlashSpec UCB1,
  γ=4): **44.2 tok/s**, α=0.75, p50=22.1ms.
  Result committed to `benchmarks/results/flashspec_ucb_tiny_llama.json`.
- Bandit regret empirically measured (T=10,000, K=3 arms, true_rates=[0.3,0.7,0.5]):
  UCB1=100.2, Thompson=18.9, Oracle=0.0. Theory bound O(√(KT log T))≈526.
  Figure committed to `paper/figures/bandit_regret.jpg`.
- Gamma sweep results committed to `benchmarks/results/gamma_sweep.csv`
  (acceptance rates genuine; throughput values reflect sampling kernel only,
  not end-to-end inference — see CSV for details).
- Draft model size sweep committed to `benchmarks/results/draft_size_sweep.csv`.
- Notebooks updated with real Colab T4 execution outputs:
  `notebooks/01_quickstart.ipynb`, `02_bandit_analysis.ipynb`,
  `03_kernel_profiling.ipynb`.
- Triton kernel profiling on Tesla T4 committed (notebook 03). Reference
  kernel: 0.07–0.72ms. Triton kernel: faster at batch=8 (1.6×), slower at
  batch=1 (0.2×) on T4 due to HBM bandwidth constraints — expected;
  H100 profiling pending.
- README updated: all real T4 measurements separated from H100 design targets;
  ⊛ symbol marks every unverified number.



### Fixed
- **Critical**: `pip install flashspec` failed completely on Windows and
  macOS with `ERROR: No matching distribution found for triton>=3.0.0` —
  Triton publishes official PyPI wheels for Linux only, but `triton>=3.0.0`
  was a hard core dependency, making the package uninstallable on those
  platforms. (Reported against v0.1.0–v0.1.2.)
  - `pyproject.toml`: moved `triton` out of `dependencies` into a new
    `gpu` extra with a `platform_system == 'Linux'` environment marker:
    `pip install flashspec[gpu]`.
  - `flashspec/kernels/__init__.py`: now imports `verify_kernel` and
    `gather_kernel` inside a `try/except ImportError`. If Triton is
    unavailable, `verify_tokens` and `gather_accepted` are bound to a
    wrapper that raises an actionable `ImportError` only when *called*
    (not at import time), explaining `pip install flashspec[gpu]`
    (Linux-only) and pointing to the cross-platform
    `flashspec.kernels._reference` module (identical numerics, verified in
    `tests/unit/test_verify_kernel.py`). Adds module-level
    `flashspec.kernels.TRITON_AVAILABLE: bool`.
  - `tests/unit/test_verify_kernel.py`: added
    `TestKernelsPackageWithoutTriton` (3 tests, CPU-only, no GPU/Triton
    required) verifying the package imports cleanly and raises the
    actionable error when Triton is absent.
  - `.github/workflows/{ci,gpu_tests,benchmark}.yml`: updated `pip install`
    steps to request the `gpu` extra wherever Triton-backed kernel tests
    or benchmarks actually run (all on `ubuntu-22.04`/self-hosted Linux
    runners, where the extra resolves normally).
  - `README.md`: rewrote the Installation section — `pip install flashspec`
    is now correctly described as cross-platform (CPU reference kernels),
    with `pip install flashspec[gpu]` for Linux+CUDA accelerated kernels.
  - Bumped version `0.1.0` → `0.1.3` in `pyproject.toml` and
    `flashspec/__init__.py` (0.1.0–0.1.2 are broken on non-Linux and
    should be marked yanked on PyPI after this release).

---
- Initial project scaffold following AGENTS.md specification.
- `flashspec.utils.config`: Pydantic v2 configuration models (`FlashSpecConfig`,
  `BanditConfig`, `SamplingConfig`, `MetricsConfig`). All immutable (`frozen=True`).
- `flashspec.utils.logging`: Structured JSON logger via `get_logger`.
- `flashspec.utils.device`: Device detection, seeding (`set_seed`), and MFU helpers.
- `flashspec.kernels._reference`: Pure-PyTorch reference implementations of
  `verify_tokens_reference` and `gather_accepted_reference` (tests use only).
- `flashspec.kernels.verify_kernel`: Triton token-verification kernel with
  `@triton.autotune` over 4 tile configs; handles float32 and bfloat16.
- `flashspec.kernels.gather_kernel`: Triton gather/scatter helpers.
- `flashspec.sampling.rejection`: Algorithm 1 (Leviathan et al. 2023) exact
  speculative sampling with correct residual distribution. Temperature applied
  by caller before log-softmax per §7.
- `flashspec.sampling.typical`: Typical-acceptance sampling variant.
- `flashspec.bandit.base`: `DraftSelector` abstract base class with JSON
  serialisation contract and windowed statistics; thread-safe via `threading.Lock`.
- `flashspec.bandit.ucb`: UCB1 selector satisfying O(√(KT log T)) regret bound.
- `flashspec.bandit.thompson`: Thompson sampling selector with Beta conjugate prior.
- `flashspec.bandit.oracle`: Oracle selector for regret upper-bound experiments.
- `flashspec.engine.drafter`: `DraftModel` protocol and entry-point registry.
- `flashspec.engine.verifier`: `TargetModel` wrapper with temperature applied
  before `log_softmax` (§7 compliance).
- `flashspec.engine.speculative`: `SpeculativeEngine` main orchestrator.
- `flashspec.metrics.acceptance`: Token acceptance rate tracker.
- `flashspec.metrics.throughput`: Tokens/s and MFU tracker.
- `flashspec.metrics.latency`: p50/p95/p99 latency tracker.
- `flashspec.export.onnx`: ONNX export for draft models (opset ≥ 17).
- `tests/unit/test_verify_kernel.py`: 13 CPU reference tests + 6 GPU Triton
  parity tests (`@pytest.mark.gpu`) + 1 Hypothesis property test.
- `tests/unit/test_sampling.py`: All §5.2 mandatory invariants + 3 Hypothesis
  property tests (shape, first_rejection range, padding invariant).
- `tests/unit/test_bandit.py`: Convergence, regret bound, serialisation,
  adversarial swap, thread-safety + 3 Hypothesis property tests.
- `tests/unit/test_acceptance.py`: Full AcceptanceTracker coverage.
- `tests/unit/test_config.py`: Pydantic validation + 3 Hypothesis property tests.
- `tests/integration/test_e2e_greedy.py`: 4 tests (greedy equiv + 3 edge cases).
- `tests/integration/test_e2e_sampling.py`: §2.1 hard gate at N=10,000 samples
  + 4 edge cases. Replaces the previous N=1,000 fast gate.
- `tests/integration/test_onnx_parity.py`: §2.4 parity for batch 1..8, batch
  9..32, variable seq_len, invalid opset edge case.
- `tests/chaos/test_bandit_adversarial.py`: 0%, 100%, many-arms, swap scenarios.
- `benchmarks/run_all.py`: §6-compliant runner — `set_num_threads(1)`, 50
  warm-up, mean±std over 200 steps, `synchronize()`, `max_memory_allocated()`.
- `benchmarks/compare_baselines.py`: FlashSpec vs Medusa vs EAGLE vs AR.
- `benchmarks/sweep_draft_sizes.py`: Acceptance rate vs parameter count ablation.
- `benchmarks/sweep_gamma.py`: γ sweep (speculation length vs throughput).
- `benchmarks/benchmark_kernels.py`: §13.5 kernel roofline benchmark (HBM
  bandwidth model, memory-efficiency ratio, per-shape JSON output).
- `benchmarks/results/baseline.json`: Stub with exact §14 JSON schema.
- `scripts/check_regression.py`: Per-model regression thresholds from §6
  (Llama-3-8B < 1.8×, Llama-3-70B < 2.2×, Mistral-7B < 1.7×, kernel < 1 ms).
- `scripts/profile_kernel.py`: Nsight/torch.profiler latency table.
- `scripts/export_draft.py`: CLI ONNX export with parity verification.
- `scripts/download_models.py`: HuggingFace Hub checkpoint downloader.
- `scripts/update_benchmark_results.py`: §14 schema validator + git commit.
- `docs/architecture.md`: Mermaid sequence + component diagrams + correctness
  guarantee section explaining Algorithm 1 distribution preservation.
- `docs/kernels.md`: Tiling strategy, SRAM analysis, roofline for both kernels.
- `docs/bandit.md`: UCB1 regret proof sketch, convergence figure, windowed
  statistics justification.
- `docs/benchmarks.md`: Reproduction guide for every paper number.
- `mkdocs.yml` at repo root: MkDocs site configuration (Material theme,
  mkdocstrings, Mermaid support).
- `.readthedocs.yaml`: Read the Docs build config (Python 3.11, ubuntu-22.04,
  MkDocs Material auto-install). Required by Read the Docs for all new projects.
- `docs/requirements.txt`: Pinned doc dependencies for reproducible RTD builds.
- `paper/flashspec.tex`: 9-section LaTeX paper (Introduction, Background,
  Method, Experiments, Ablation, Analysis, Related Work, Conclusion,
  Reproducibility Statement).
- `paper/flashspec.bib`: All 10 references (Leviathan, Auer, Thompson, Cai,
  Li, Chen, Chapelle, Dao, Triton, Mattral self-citation).
- `paper/figures/.gitkeep`: Vector-figure output directory for the paper
  (PDF/SVG only, generated by notebooks 02/03).
- `notebooks/01_quickstart.ipynb`: Colab-ready quickstart, < 10 min runtime.
- `notebooks/02_bandit_analysis.ipynb`: Regret curve plots → `paper/figures/`.
- `notebooks/03_kernel_profiling.ipynb`: Reference vs Triton latency table.
- `.github/workflows/ci.yml`: 6-job pipeline in exact §12.1 order — lint
  (+ pip-audit CRITICAL CVE gate), test-cpu, test-chaos, coverage (≥ 95%),
  onnx-parity, docs-build. All jobs on ubuntu-22.04 + TRITON_CPU_BACKEND=1.
- `.github/workflows/gpu_tests.yml`: test-gpu (A10G, N=10,000 KS gate) +
  bench-quick (fails if throughput < 1.5× AR).
- `.github/workflows/benchmark.yml`: bench-full (H100) → Trivy image scan →
  `update_benchmark_results.py` → git commit + push → Slack alert on failure.
- `CONTRIBUTING.md`: Commit format, branching, PR rules, autonomous vs
  human-approval boundaries (§9, §13.1, §13.2).
- `deploy/Dockerfile`: CUDA 12.4 image.
- `deploy/docker-compose.yml` and `deploy/k8s/flashspec-deployment.yaml`.
- `social/x_thread.md`: X (Twitter) launch thread (long + short variants).
- `social/linkedin_post.md`: LinkedIn launch post with engineering narrative.
- `paper/joss/paper.md` + `paper/joss/paper.bib`: JOSS submission (673 words,
  within the 250–1000 word requirement; all citation keys verified present).
- `paper/joss/README.md`: JOSS submission checklist and step-by-step guide.
- `CITATION.cff`: GitHub-native citation metadata (renders "Cite this
  repository" button); includes `preferred-citation` block.
- `.zenodo.json`: Zenodo archival metadata, activates on first GitHub Release.
- `PUBLISHING.md`: Master guide sequencing GitHub Release → Zenodo → JOSS →
  arXiv → social posts, with clear blockers identified per deliverable.

### Fixed
- `engine/speculative.py`: Renamed local variable `tps` → `tokens_per_second`
  (§3.4 naming convention violation).
- `engine/drafter.py`: Replaced bare `except Exception: return` with logged
  exception (§13.6 banned silent-except pattern).
- `sampling/rejection.py`: Removed unused `temperature` parameter — temperature
  is the caller's responsibility per §7.
- `engine/verifier.py`: Applied temperature scaling to raw logits **before**
  `log_softmax` (§7 compliance). Added `temperature` parameter to `score_draft`.
- `engine/speculative.py`: Passes `temperature` to `score_draft` instead of
  (incorrectly) forwarding it to `rejection_sample`.
- `engine/verifier.py`: Fixed two lines exceeding 100 characters (§3.1);
  `# type: ignore` annotations now carry their explanatory comment on the
  same line as required by §13.2.
- `metrics/throughput.py`: Replaced `tps` with `tokens_per_second` in
  docstring example (§3.4).
- `benchmarks/results/baseline.json`: Removed non-spec `_note` field; schema
  now matches §14 exactly.
- `benchmarks/run_all.py`: Removed non-spec `schema_version` from result dict;
  added `torch.cuda.synchronize()` around all timed regions (§6).
- `scripts/update_benchmark_results.py`: Removed `schema_version` from required
  keys to match §14 schema.
- `tests/integration/test_e2e_sampling.py`: Upgraded CI hard gate from N=1,000
  to N=10,000 (§2.1 requirement). Added 4 edge cases.
- `tests/unit/test_acceptance.py`, `tests/unit/test_bandit.py`,
  `tests/unit/test_config.py`: Added failure messages to all bare `assert`
  statements (§5.3); `test_config.py` rewritten with 3 Hypothesis property tests.
- `docs/mkdocs.yml`: Moved from `docs/` to repo root (MkDocs requirement).
  `docs/mkdocs.yml` stub retained per §1 layout tree.
- `flashspec/kernels/_reference.py`: Fixed a 102-character line (§3.1).
- `AGENTS.md`: Copied into the repository root (§1 — was missing).
- `paper/flashspec.bib`: **Critical** — repaired a malformed `triton2019`
  entry left over from a prior edit. The `@inproceedings{triton2019,` header
  and `title` field had been accidentally deleted, leaving orphaned fields
  (`author`, `booktitle`, `year`) with no entry header — this would have
  caused `bibtex` to fail and `make paper` to error out. Verified post-fix:
  10 entries, balanced braces, all 9 `\cite{}` keys in `flashspec.tex`
  resolve correctly (`mattral2025flashspec` is defined-but-unused by design —
  it is the self-citation for external use, not cited within the paper).

---

## [0.1.0] — 2025-11-01

- Initial release (scaffold only; no real model weights required).
