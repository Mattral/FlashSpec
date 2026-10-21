# FlashSpec

**Adaptive speculative-decoding inference engine with Triton-optimised
verification and online bandit draft selection.**

---

## Navigation

- **[Architecture](architecture.md)** — system design, sequence diagram, correctness guarantee
- **[Kernels](kernels.md)** — Triton kernel tiling, SRAM analysis, H100 roofline arithmetic (not a profile)
- **[Bandit](bandit.md)** — UCB1/Thompson, regret bound, windowed statistics
- **[Benchmarks](benchmarks.md)** — committed results, targets, and how a run is recorded
- **[Postponed](postponed.md)** — packaging and GPU measurements that are not blockers
- **[API Reference](api/index.md)** — auto-generated from docstrings

## Quick links

- [GitHub repository](https://github.com/Mattral/FlashSpec)
- [Zenodo record (June 2026)](https://doi.org/10.5281/zenodo.20766119)
- Draft preprint: `paper/flashspec.tex` (October 2026). Not deposited and not on arXiv.
- [Benchmark results](benchmarks.md)
