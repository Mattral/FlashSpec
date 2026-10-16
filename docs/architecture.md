# FlashSpec Architecture

## Overview

FlashSpec is an adaptive speculative-decoding inference engine with three
orthogonal innovations over prior work:

1. **Triton-optimised verification** — batched token acceptance in a single
   kernel call; O(SRAM) independent of vocabulary size.
2. **Online bandit draft selection** — UCB1 or Thompson sampling continuously
   adapts which draft model is used, maximising acceptance rate without human tuning.
3. **Exact output distribution** — Algorithm 1 (Leviathan et al., 2023) with the
   correct residual distribution. The check is the chi-squared test in
   `tests/integration/test_output_distribution.py`. The lint job is off, so
   the CI jobs that depend on it do not run that test.

---

## Sequence diagram — speculative decoding loop

```mermaid
sequenceDiagram
    participant C as Caller
    participant E as SpeculativeEngine
    participant B as Bandit (UCB1/Thompson)
    participant D as DraftModel
    participant T as TargetModel
    participant K as verify_tokens (Triton)
    participant S as rejection_sample

    C->>E: generate(input_ids, max_new_tokens)
    loop until max_new_tokens
        E->>B: select() → arm index
        E->>D: generate_draft(ctx, γ) → draft_ids, draft_logprobs
        E->>T: score_draft(ctx, draft_ids, γ) → target_logprobs
        E->>K: verify_tokens(draft_lp, target_lp, u) → accepted, first_rejection
        E->>S: rejection_sample(...) → accepted_ids, α
        E->>B: update(arm, n_accepted)
        E->>E: ctx = cat(ctx, accepted_ids)
    end
    E-->>C: GenerationResult(output_ids, α, tokens_per_second)
```

---

## Batching

`SpeculativeEngine.generate` keeps one context tensor per row. When those
lengths differ, the loop groups the active rows by context length and
forwards only the largest group. Other lengths run on a later iteration.
There is one bandit choice per loop.

The drafter and the target roll their KV caches only when the group that
runs has one row (`len(idxs) == 1` in `flashspec/engine/speculative.py`).
A larger group full-forwards. A model that cannot continue from
`past_key_values` disables the cache and stays on the full forward.
The cache is not a padded multi-row batch.

The returned `output_ids` tensor is rectangular. A row that produced fewer
than `max_new_tokens` ids is filled with `-1`. Callers drop those
positions. The fill is not an attention mask. This engine does not pad
unequal prompts into one batch and does not build a padding mask.

---

## Component diagram — module dependencies

```mermaid
graph TD
    U[utils<br/>config · logging · device]
    K[kernels<br/>verify_kernel · gather_kernel<br/>_reference]
    SA[sampling<br/>rejection · typical]
    BA[bandit<br/>UCB1 · Thompson · Oracle]
    ME[metrics<br/>acceptance · throughput · latency]
    EX[export<br/>onnx]
    EN[engine<br/>drafter · verifier · SpeculativeEngine]

    U --> K
    U --> SA
    U --> BA
    U --> ME
    U --> EX
    U --> EN
    K --> SA
    K --> EN
    SA --> EN
    BA --> EN
    ME --> EN
    EX --> EN

    style K fill:#f9a,stroke:#c66
    style EN fill:#adf,stroke:#36a
    style BA fill:#afa,stroke:#393
```

Import direction is strictly bottom-up: `utils → kernels → sampling → bandit → metrics → export → engine`.
No layer may import from a layer above it.  Enforced by `import-linter` in CI.

---

## Module hierarchy

```
flashspec/
├── utils/          Low-level: config (Pydantic v2), logging (JSON), device
├── kernels/        Triton verify_kernel, gather_kernel; _reference (tests only)
├── sampling/       rejection.py (Algorithm 1), typical.py
├── bandit/         base.py (ABC), ucb.py, thompson.py, oracle.py
├── metrics/        acceptance.py, throughput.py, latency.py
├── export/         onnx.py
└── engine/         drafter.py (protocol + registry), verifier.py, speculative.py
```

---

## Correctness guarantee

**Theorem** (Leviathan et al., 2023, Theorem 1): The output distribution of
`SpeculativeEngine.generate()` is identical to autoregressive sampling from
the target model `p`, regardless of the draft model `q`.

**Why?** At each draft position `i`:

1. The draft token `x_i ~ q(· | ctx)` is accepted with probability
   `min(1, p(x_i) / q(x_i))`.
2. If rejected, a residual token is sampled from the adjusted distribution
   `max(0, p - q) / ||max(0, p - q)||₁`.

The resulting marginal distribution over accepted tokens is exactly `p`.

**Implementation invariants in this codebase:**

- The residual distribution is computed as
  `torch.clamp(p - q, min=0.0) / denom.clamp(min=1e-9)` — no temperature,
  no softmax applied to the residual (verified by `test_sampling.py`).
- All acceptance-probability comparisons operate in log-space:
  `accept_prob = exp(log_p - log_q).clamp(max=1.0)` — never `p / q` directly.
- The Triton kernel and pure-PyTorch reference produce identical boolean masks
  to within `atol=1e-5` (float32) / `atol=1e-3` (bfloat16), verified by
  `tests/unit/test_verify_kernel.py`.
- `tests/integration/test_output_distribution.py` is the chi-squared check
  that emitted tokens follow `p` when `q` differs, at gamma 1 and 4
  (N=30,000, vocabulary 6, df=5, 1% point 15.086), plus one toy target
  whose next token depends on the previous token.
  `tests/integration/test_e2e_sampling.py` only covers draft equal to
  target, where every gamma-1 draft token is accepted.

---

## Expected tokens per target forward

With a constant acceptance rate α and speculation length γ, the expected
number of tokens per target forward is

```
(1 - α^(γ+1)) / (1 - α)
```

At α = 0.75 and γ = 4 this is about 3.05. The product γ·α+1 is a different
approximation and is not this expectation. The figure is the expectation
under that constant rate. It is not a measured speedup. The committed CPU
and Tesla T4 pairs are slower than autoregressive decoding.

---

## References

1. Leviathan et al. (2023), "Fast Inference from Transformers via Speculative
   Decoding", arXiv:2211.17192 — Algorithm 1, Theorem 1.
2. Auer et al. (2002), *Machine Learning* 47(2-3) — UCB1 regret bound.
3. Cai et al. (2024), arXiv:2401.10774 — Medusa (baseline).
4. Li et al. (2024), arXiv:2401.15077 — EAGLE (baseline).
