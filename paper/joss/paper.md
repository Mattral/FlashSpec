---
title: 'FlashSpec: An Adaptive Speculative Decoding Engine with GPU-Native
  Verification and Online Bandit Draft Selection'
tags:
  - Python
  - machine learning
  - large language models
  - GPU computing
  - Triton
  - inference optimization
  - multi-armed bandits
authors:
  - name: Min Htet Myet
    orcid: 0009-0005-0853-8301
date: 2026-10-16
bibliography: paper.bib
---

# Summary

`FlashSpec` is an open-source Python library implementing adaptive
speculative decoding for large language model (LLM) inference. Speculative
decoding accelerates autoregressive generation by having a small "draft"
model propose several candidate tokens, which a larger "target" model then
verifies in a single forward pass, accepting or rejecting each candidate
according to a probabilistic criterion that preserves the target model's
output distribution exactly [@leviathan2023fast].

`FlashSpec` provides two components. First, the token-level verification step —
the comparison of draft and target log-probabilities that decides
acceptance — is implemented as a Triton GPU kernel and as a pure-PyTorch
reference. The kernel reads only the two scalar log-probabilities relevant
to each candidate token, giving it a memory footprint that is independent of
vocabulary size. A Tesla T4 measurement at batch 1 found the Triton kernel
slower than the reference. Second, `FlashSpec` frames
the choice of *which* draft model to use at each decoding step as a
K-armed bandit problem, solved online with either UCB1 [@auer2002finite] or
Thompson sampling [@thompson1933; @chapelle2011thompson]. This allows the
system to adapt to shifting acceptance rates across prompt types (e.g.,
conversational text versus source code) without manual retuning.

The library is implemented as a typed Python package (`flashspec`) with
Pydantic-based configuration, a structured logging layer, and a modular
architecture separating kernels, sampling algorithms, bandit selectors,
metrics, and the orchestration engine. It ships with a test suite: unit tests, integration tests, property-based
tests via Hypothesis, adversarial "chaos" tests for the bandit components,
and a chi-squared check that emitted tokens follow the target distribution
when the draft and target probabilities differ. The CPU test job runs that
check. The lint job runs `ruff` and import-linter. `mypy --strict` is not
part of the green lint job.

# Statement of need

Speculative decoding has become a standard technique for reducing LLM
inference latency, with production systems and research frameworks such as
Medusa [@cai2024medusa] and EAGLE [@li2024eagle] building on the original
formulation [@leviathan2023fast]. This repository studies two design choices.

First, the acceptance/rejection test is implemented here as a Triton kernel
and as a pure-PyTorch reference. The measured Tesla T4 batch-1 comparison
found the Triton kernel slower than the reference. That timing is a kernel
comparison, separate from end-to-end generation. Second, the choice
of draft model is treated as a static, offline hyperparameter, despite
empirical evidence that token acceptance rates vary substantially with
prompt domain and context length — meaning a drafter tuned for one workload
may be suboptimal for another within the same deployment.

`FlashSpec` puts both choices in one testable codebase. The GPU-resident
verification kernel keeps the accept/reject test on device. The measured
batch-1 comparison did not show it to be faster than the PyTorch reference.
The bandit-based selector removes the need for manual
drafter selection and provides a formal regret guarantee
(O(√(KT log T)) for UCB1) governing how quickly the system converges to the
best-performing drafter for a given workload. Researchers studying
speculative decoding, practitioners deploying LLM inference services, and
students learning GPU kernel programming with Triton may find `FlashSpec`
useful as a readable implementation of those two choices. The project
status is alpha. The committed generation pairs are slower than
autoregressive decoding. The coverage floor is the measured
unit-and-integration coverage, rounded down.

# Implementation and correctness

A central design goal of `FlashSpec` is that adaptive draft selection and
GPU-resident verification must not alter the statistical distribution of
generated tokens relative to standard autoregressive sampling from the
target model. This is checked by a chi-squared test on discrete token
ids (α = 0.01, df = 5, N = 30,000, vocabulary 6) for fixed draft and
target distributions that differ, at gamma 1 and gamma 4, including the
second emitted token. A separate test uses a toy target whose next token
depends on the previous token. The gamma 1 case with identical draft and
target only checks that every draft token is accepted. Numerical parity
between the Triton kernel and
a pure-PyTorch reference implementation is verified separately for both
float32 (tolerance 1e-5) and bfloat16 (tolerance 1e-3) inputs.

The bandit selectors (UCB1, Thompson sampling, and an oracle baseline used
only for regret-bound experiments) support JSON serialization for
checkpointing, windowed statistics for tracking non-stationary acceptance
rates, and thread-safe updates. New draft models can be registered via a
Python entry-point mechanism without modifying library internals.

Committed end-to-end pairs are slower than autoregressive decoding on the
same prompts. A CPU float32 HumanEval/23 run measured 1.75 tok/s against
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
