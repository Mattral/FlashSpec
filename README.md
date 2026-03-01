# FlashSpec

**Adaptive speculative-decoding inference engine with Triton-optimised
verification and online bandit draft selection.**

[![CI](https://github.com/Mattral/FlashSpec/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/FlashSpec/actions/workflows/ci.yml)
[![GPU Tests](https://github.com/Mattral/FlashSpec/actions/workflows/gpu_tests.yml/badge.svg)](https://github.com/Mattral/FlashSpec/actions/workflows/gpu_tests.yml)
[![codecov](https://codecov.io/gh/Mattral/FlashSpec/branch/main/graph/badge.svg)](https://codecov.io/gh/Mattral/FlashSpec)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

---

## Key innovations

| Innovation | Description |
|---|---|
| **Triton verification** | Accept/reject in a single kernel call; O(SRAM) in vocab size |
| **Online bandit selection** | UCB1/Thompson continuously adapts the draft model; O(√KT log T) regret |
| **Exact distribution** | Algorithm 1 (Leviathan et al. 2023) with correct residual; KS-verified on every CI run |

---

## Quickstart

```python
from flashspec import FlashSpecConfig, SpeculativeEngine, BanditConfig, SamplingConfig
from flashspec.bandit import UCB1Selector

config = FlashSpecConfig(
    drafter_name="llama3-1b",
    target_name="meta-llama/Llama-3-8B-Instruct",
    device="cuda:0",
    dtype="bfloat16",
    max_new_tokens=256,
    bandit=BanditConfig(n_arms=2, strategy="ucb1"),
    sampling=SamplingConfig(gamma=4),
)

# engine = SpeculativeEngine(config, drafters=[...], target=..., bandit=UCB1Selector(n_arms=2))
# result = engine.generate(input_ids)
# print(f"α={result.acceptance_rate:.2f}  {result.tokens_per_second:.0f} tok/s")
```

See `notebooks/01_quickstart.py` for a runnable CPU demo.

---

## Installation

```bash
# CPU-only (no Triton):
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.4):
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2
- Triton ≥ 3.0 (GPU only)
- CUDA ≥ 12.0 (GPU only)

---

## Repository structure

```
flashspec/          Core library
  utils/            Config (Pydantic v2), logging (JSON), device
  kernels/          Triton verify + gather; _reference (tests only)
  sampling/         rejection.py (Alg. 1), typical.py
  bandit/           ucb.py, thompson.py, oracle.py
  metrics/          acceptance, throughput, latency
  export/           ONNX
  engine/           drafter registry, verifier, SpeculativeEngine

tests/
  unit/             ≥ 95% line coverage
  integration/      KS-test distribution gate, ONNX parity
  chaos/            Adversarial bandit scenarios

benchmarks/         Configs, run_all.py, sweep_gamma.py, baselines.py
docs/               Architecture, kernels, bandit, benchmarks
paper/              LaTeX source (MLSys 2026 / NeurIPS 2025 target)
deploy/             Dockerfile, docker-compose.yml, k8s/
```

---

## Running tests

```bash
# Fast CPU tests (no GPU required):
make test

# GPU tests (requires CUDA device):
make test-gpu

# Chaos / adversarial tests:
make test-chaos

# Toy benchmark (no model weights):
make bench-quick
```

---

## Performance targets (Llama-3-8B-Instruct, γ=4, H100)

| Metric | Target |
|--------|--------|
| Speedup vs vanilla AR | ≥ 2.0× |
| Mean acceptance rate α | ≥ 0.65 |
| p99 step latency | ≤ 50 ms |

---

## Citation

```bibtex
@article{myet2025flashspec,
  title   = {{FlashSpec}: Adaptive Speculative Decoding with Online Bandit
             Draft Selection and {Triton}-Optimised Verification},
  author  = {Myet, Min Htet},
  year    = {2025},
  url     = {https://github.com/Mattral/FlashSpec},
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
