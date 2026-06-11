# Changelog

All notable changes to FlashSpec are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/).

---

## [Unreleased]

### Added
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
  warm-up, mean±std over 200 steps, `max_memory_allocated()`.
- `benchmarks/compare_baselines.py`: FlashSpec vs Medusa vs EAGLE vs AR.
- `benchmarks/sweep_draft_sizes.py`: Acceptance rate vs parameter count ablation.
- `benchmarks/sweep_gamma.py`: γ sweep (speculation length vs throughput).
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
- `paper/flashspec.tex`: 10-section LaTeX skeleton (Introduction, Background,
  Method, Experiments, Ablation, Analysis, Related Work, Conclusion,
  Reproducibility Statement).
- `paper/flashspec.bib`: All references (Leviathan, Auer, Thompson, Cai, Li,
  Chen, Chapelle, Dao, Triton).
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
- `metrics/throughput.py`: Replaced `tps` with `tokens_per_second` in
  docstring example (§3.4).
- `benchmarks/results/baseline.json`: Removed non-spec `_note` field; schema
  now matches §14 exactly.
- `benchmarks/run_all.py`: Removed non-spec `schema_version` from result dict.
- `scripts/update_benchmark_results.py`: Removed `schema_version` from required
  keys to match §14 schema.
- `tests/integration/test_e2e_sampling.py`: Upgraded CI hard gate from N=1,000
  to N=10,000 (§2.1 requirement). Added 4 edge cases.
- `docs/mkdocs.yml`: Moved from `docs/` to repo root (MkDocs requirement).
  `docs/mkdocs.yml` stub retained per §1 layout tree.

---


## [0.1.1] - 2026-06-12
### Fixed
- Corrected inline `# type: ignore` annotations in `engine/verifier.py` (≤100 chars, explanation inline).
- Added mandatory `mattral2025flashspec` BibTeX entry to `paper/flashspec.bib`.

### Documentation
- All docs audited and updated to v0.2.2 spec compliance.


## [0.1.0] — 2025-11-01

- Initial release (scaffold only; no real model weights required).
