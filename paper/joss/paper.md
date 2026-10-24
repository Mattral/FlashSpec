---
title: 'FlashSpec: Lossless Speculative Decoding with a Static-Cache
  CUDA-Graph Fast Path'
tags:
  - Python
  - machine learning
  - large language models
  - GPU computing
  - speculative decoding
  - inference optimization
authors:
  - name: Min Htet Myet
    orcid: 0009-0005-0853-8301
date: 2026-10-16
bibliography: paper.bib
---

# Summary

`FlashSpec` is an open-source Python library implementing speculative
decoding for HuggingFace causal language models (LLMs). Speculative
decoding accelerates autoregressive generation by having a small "draft"
model propose several candidate tokens, which a larger "target" model then
verifies in a single forward pass, accepting or rejecting each candidate
according to a probabilistic criterion that preserves the target model's
output distribution exactly [@leviathan2023fast].

Each batch-1 step runs one target forward over the pending token and the
draft tokens, decides acceptance on the device in PyTorch (rejection
sampling at temperature > 0, the longest argmax-matching prefix at
temperature 0), and reads one small tensor back to the host. On CUDA, an
optional compiled path keeps the drafter and target on static KV caches
and replays the fixed-shape decode and verify calls as CUDA graphs.

Two further components are experimental. A UCB1 [@auer2002finite] or
Thompson-sampling [@thompson1933; @chapelle2011thompson] selector can
choose between registered drafters, but every measured run used one
drafter. A Triton verification kernel and an optional assembly residual
exist with parity tests; the engine does not call them, and they are not
on the generation critical path. A Tesla T4 measurement at batch 1 found
the Triton kernel slower than the PyTorch reference.

The library is implemented as a typed Python package (`flashspec`) with
Pydantic-based configuration, a structured logging layer, and a modular
architecture separating kernels, sampling algorithms, bandit selectors,
metrics, and the orchestration engine. It ships with a test suite: unit tests, integration tests, property-based
tests via Hypothesis, adversarial "chaos" tests for the bandit components,
and a chi-squared check that emitted tokens follow the target distribution
when the draft and target probabilities differ. The CPU test job runs that
check. The lint job runs `ruff`, import-linter and `mypy --strict`.

# Statement of need

Speculative decoding has become a standard technique for reducing LLM
inference latency, with production systems and research frameworks such as
Medusa [@cai2024medusa] and EAGLE [@li2024eagle] building on the original
formulation [@leviathan2023fast]. Whether speculative decoding is faster
than plain decoding depends on the step structure (how many target and
drafter forwards a step costs), on host synchronisation, and on kernel
launch overhead at batch 1, as much as on the acceptance rate.

`FlashSpec` is a small, tested implementation in which those costs are
visible: the benchmark harness records forward counts per step, host
reads per step, a per-step time breakdown, and the drafter-to-target cost
ratio, next to HuggingFace `generate`, `generate` on a compiled static
cache, and HuggingFace assisted generation on the same prompts. Researchers
and students studying speculative decoding may find it useful as a
readable implementation with lossless-output tests. The project status is
alpha. The coverage floor is the measured unit-and-integration coverage,
rounded down.

# Implementation and correctness

A central design goal of `FlashSpec` is that speculative decoding must not
alter the statistical distribution of generated tokens relative to
standard autoregressive sampling from the target model. This holds for the
`rejection` and `greedy` variants; the optional `typical` variant is
lossy and labelled as such. This is checked by a chi-squared test on discrete token
ids (α = 0.01, df = 5, N = 30,000, vocabulary 6) for fixed draft and
target distributions that differ, at gamma 1 and gamma 4, including the
second emitted token. A separate test uses a toy target whose next token
depends on the previous token. The gamma 1 case with identical draft and
target only checks that every draft token is accepted. Greedy output is
compared token-for-token with `generate(do_sample=False)` on CPU. Numerical
parity between the experimental Triton kernel and a pure-PyTorch reference
is checked separately (float32 tolerance 1e-5, bfloat16 1e-3) by GPU tests
that run by hand on Colab.

The experimental bandit selectors (UCB1, Thompson sampling, and an oracle
baseline used only for regret-bound experiments) support JSON serialization for
checkpointing, windowed statistics for tracking non-stationary acceptance
rates, and thread-safe updates. New draft models can be registered via a
Python entry-point mechanism without modifying library internals.

What has been measured is narrow. On one model pair
(Qwen2.5-1.5B-Instruct verified by Qwen2.5-0.5B-Instruct), one Tesla T4,
fp16, batch 1, 10 MT-Bench and 10 HumanEval prompts, with the compiled path
on, `FlashSpec` was faster than HuggingFace `generate` compiled the same
way in all four dataset and decoding-mode cells; the values are generated
from the result files into the repository README. With the eager path, it
was slower than plain `generate` in 7 of 8 cells of the earlier run on
two pairs. Every run used one drafter, so the bandit was not exercised.

Earlier pairs from the 0.1.5 engine were slower than autoregressive
decoding on the same prompts. A CPU float32 HumanEval/23 run measured 1.75 tok/s against
4.30 tok/s (acceptance 0.114, speedup 0.407). A Colab Tesla T4 float16
run of eight synthetic samples measured 17.6 tok/s against 29.0 tok/s
(speedup 0.606). An older notebook cell assigns
44.2 tok/s, acceptance 0.75, p50 22.1 ms; no timing log exists, so it
is not used as a result. A synthetic three-arm simulation reports UCB1
regret 100.2 over T = 10,000 rounds, inside a bound near 526. That
simulation is not language-model throughput. Medusa and EAGLE are not
implemented, and no H100 generation result is committed. The longer
research draft is `paper/flashspec.tex`. It is not deposited and it is
not under review.

# Acknowledgements

The author thanks the open-source PyTorch and OpenAI Triton communities,
whose tooling makes projects like this practical to build and verify.

# References
