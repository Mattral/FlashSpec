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
  `BanditConfig`, `SamplingConfig`, `MetricsConfig`).
- `flashspec.utils.logging`: Structured JSON logger via `get_logger`.
- `flashspec.utils.device`: Device detection and seeding utilities.
- `flashspec.kernels._reference`: Pure-PyTorch reference implementations of
  `verify_tokens_reference` and `gather_accepted_reference`.
- `flashspec.kernels.verify_kernel`: Triton token-verification kernel with
  `@triton.autotune` over 4 tile configs; handles float32 and bfloat16.
- `flashspec.kernels.gather_kernel`: Triton gather/scatter helpers.
- `flashspec.sampling.rejection`: Algorithm 1 (Leviathan et al. 2023) exact
  speculative sampling with correct residual distribution.
- `flashspec.sampling.typical`: Typical-acceptance sampling variant.
- `flashspec.bandit.base`: `DraftSelector` abstract base class with JSON
  serialisation contract and windowed statistics support.
- `flashspec.bandit.ucb`: UCB1 selector satisfying O(√(KT log T)) regret bound.
- `flashspec.bandit.thompson`: Thompson sampling selector with Beta conjugate prior.
- `flashspec.bandit.oracle`: Oracle selector for regret upper-bound experiments.
- `flashspec.engine.drafter`: `DraftModel` protocol and entry-point registry.
- `flashspec.engine.verifier`: `TargetModel` wrapper.
- `flashspec.engine.speculative`: `SpeculativeEngine` main orchestrator.
- `flashspec.metrics.acceptance`: Token acceptance rate tracker.
- `flashspec.metrics.throughput`: Tokens/s and MFU tracker.
- `flashspec.metrics.latency`: p50/p95/p99 latency tracker.
- `flashspec.export.onnx`: ONNX export for draft models.
- `tests/`: Full unit, integration, and chaos test suites.
- `benchmarks/`: Benchmark runner, baseline comparison, and sweep scripts.
- `benchmarks/configs/`: YAML configs for Llama-3-8B, Llama-3-70B, Mistral-7B.
- `docs/`: MkDocs architecture, kernels, bandit, and benchmarks documentation.
- `paper/flashspec.tex`: MLSys 2026 / NeurIPS 2025 LaTeX source skeleton.
- CI/CD GitHub Actions workflows: `ci.yml`, `gpu_tests.yml`, `benchmark.yml`.
- `deploy/Dockerfile`, `docker-compose.yml`, `k8s/flashspec-deployment.yaml`.

---

## [0.1.0] — 2025-11-01

- Initial release (scaffold only; no real model weights required).
