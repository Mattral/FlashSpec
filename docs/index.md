# FlashSpec

**Lossless speculative decoding for HuggingFace causal language models,
with a static-cache CUDA-graph fast path.**

Measured on one model pair, one Tesla T4, batch 1, with compile on; see
[Benchmarks](benchmarks.md) and the README tables. The bandit draft
selector is experimental (measured once over seven arms, in Phase 6, where
it did not beat the best fixed arm), and the Triton and assembly kernels
are experimental and not on the generation critical path.

---

## Navigation

- **[Install](install.md)** — every install route, platform notes, requirements
- **[Examples](examples.md)** — a CPU example with `load_drafters`; the README has the quickstart and recipes
- **[Architecture](architecture.md)** — system design, sequence diagram, correctness guarantee
- **[Kernels](kernels.md)** — experimental Triton kernels (not called by the engine): tiling, SRAM analysis, H100 roofline arithmetic (not a profile)
- **[Bandit](bandit.md)** — experimental UCB1/Thompson selector, regret bound, windowed statistics
- **[Benchmarks](benchmarks.md)** — committed results, targets, and how a run is recorded
- **[Development](development.md)** — tests, repository layout, links
- **[Postponed](postponed.md)** — packaging and GPU measurements that are not blockers
- **[API Reference](api/index.md)** — auto-generated from docstrings

## Quick links

- [GitHub repository](https://github.com/Mattral/FlashSpec)
- [Zenodo record (June 2026)](https://doi.org/10.5281/zenodo.20766119)
- Draft preprint: `paper/flashspec.tex` (October 2026). Not deposited and not on arXiv.
- [Benchmark results](benchmarks.md)
- [llms.txt](https://github.com/Mattral/FlashSpec/blob/main/llms.txt) — an index of these docs for AI agents ([llmstxt.org](https://llmstxt.org) format); the built site serves a copy at `/llms.txt`
