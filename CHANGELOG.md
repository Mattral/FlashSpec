# Changelog

All notable changes to FlashSpec are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/).

---

## [Unreleased]

### Added
- First real GPU measurements (Tesla T4, TinyLlama-1.1B-Chat NF4, FlashSpec UCB1,
  γ=4): **44.2 tok/s**, α=0.75, p50=22.1ms.
  Result in `benchmarks/results/flashspec_ucb_tiny_llama.json`.
- Bandit regret measured empirically (T=10,000, K=3 arms): UCB1=100.2,
  Thompson=18.9, Oracle=0.0. Figure saved to `paper/figures/bandit_regret.jpg`.
- Gamma sweep results in `benchmarks/results/gamma_sweep.csv` (synthetic
  logprobs, acceptance rates genuine).
- Draft size sweep results in `benchmarks/results/draft_size_sweep.csv`.
- Updated `notebooks/01_quickstart.ipynb`, `02_bandit_analysis.ipynb`, and
  `03_kernel_profiling.ipynb` with real Colab T4 execution outputs.
- Kernel profiling table (T4): reference 0.07–0.72ms; Triton kernel faster
  at batch=8 (1.6×) but slower at batch=1 on T4 (expected — T4 is
  bandwidth-constrained, H100 profiling pending).
- README updated: real measurements clearly separated from H100 design targets;
  ⊛ symbol marks all unverified target numbers.

---

## [0.1.3] — Critical packaging fix (Windows/macOS install)

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
