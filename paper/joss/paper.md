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
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: TODO — set submission date
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

`FlashSpec` provides two components not commonly found together in existing
open-source implementations. First, the token-level verification step —
the comparison of draft and target log-probabilities that decides
acceptance — is implemented as a Triton GPU kernel that operates on-device,
avoiding the CPU round-trip and associated pipeline stalls present in many
reference implementations. The kernel reads only the two scalar
log-probabilities relevant to each candidate token, giving it a memory
footprint that is independent of vocabulary size. Second, `FlashSpec` frames
the choice of *which* draft model to use at each decoding step as a
K-armed bandit problem, solved online with either UCB1 [@auer2002finite] or
Thompson sampling [@thompson1933; @chapelle2011thompson]. This allows the
system to adapt to shifting acceptance rates across prompt types (e.g.,
conversational text versus source code) without manual retuning.

The library is implemented as a typed Python package (`flashspec`) with
Pydantic-based configuration, a structured logging layer, and a modular
architecture separating kernels, sampling algorithms, bandit selectors,
metrics, and the orchestration engine. It ships with a full automated test
suite — unit tests, integration tests, property-based tests via
Hypothesis, and adversarial "chaos" tests for the bandit components — along
with continuous integration workflows that run linting, type checking,
security scanning (`pip-audit`, Trivy), and a statistical distribution-
equivalence test based on the Kolmogorov–Smirnov statistic.

# Statement of need

Speculative decoding has become a standard technique for reducing LLM
inference latency, with production systems and research frameworks such as
Medusa [@cai2024medusa] and EAGLE [@li2024eagle] building on the original
formulation [@leviathan2023fast]. However, two practical gaps remain in
widely-used implementations.

First, the acceptance/rejection test is frequently implemented in a way
that requires synchronizing device and host memory at every decoding step,
introducing latency proportional to vocabulary size that scales poorly for
modern LLMs with vocabularies of 32,000 tokens or more. Second, the choice
of draft model is treated as a static, offline hyperparameter, despite
empirical evidence that token acceptance rates vary substantially with
prompt domain and context length — meaning a drafter tuned for one workload
may be suboptimal for another within the same deployment.

`FlashSpec` addresses both gaps within a single, testable codebase. The
GPU-resident verification kernel removes the host-device synchronization
bottleneck, while the bandit-based selector removes the need for manual
drafter selection and provides a formal regret guarantee
(O(√(KT log T)) for UCB1) governing how quickly the system converges to the
best-performing drafter for a given workload. Researchers studying
speculative decoding, practitioners deploying LLM inference services, and
students learning GPU kernel programming with Triton may find `FlashSpec`
useful both as a working system and as a reference implementation with an
unusually extensive test and documentation suite for a project of its
scope.

# Implementation and correctness

A central design goal of `FlashSpec` is that adaptive draft selection and
GPU-resident verification must not alter the statistical distribution of
generated tokens relative to standard autoregressive sampling from the
target model. This is verified continuously: the test suite includes a
Kolmogorov–Smirnov goodness-of-fit test (significance level α = 0.01,
N = 10,000 samples) comparing the output distribution of the speculative
sampler against the target model's distribution, run as a required check
in continuous integration. Numerical parity between the Triton kernel and
a pure-PyTorch reference implementation is verified separately for both
float32 (tolerance 1e-5) and bfloat16 (tolerance 1e-3) inputs.

The bandit selectors (UCB1, Thompson sampling, and an oracle baseline used
only for regret-bound experiments) support JSON serialization for
checkpointing, windowed statistics for tracking non-stationary acceptance
rates, and thread-safe updates. New draft models can be registered via a
Python entry-point mechanism without modifying library internals.

# Acknowledgements

The author thanks the open-source PyTorch and OpenAI Triton communities,
whose tooling makes projects like this practical to build and verify.

# References
