# FlashSpec Triton Kernels (experimental)

> **Not on the critical path.** `SpeculativeEngine` does not call these
> kernels; its accept/reject test is PyTorch on the device. A step's time is
> the drafter and target forwards, and the test is microseconds next to
> them. The kernels are kept as a systems exercise with parity tests.

## 2.1 Token-verification kernel (`verify_kernel.py`)

### Problem

Given draft log-probs `q(x | ctx)` and target log-probs `p(x | ctx)` for
γ draft positions, decide whether to accept each token.

The acceptance criterion (Leviathan et al. Algorithm 1):

```
accept_i  =  u_i  <  min(1, exp(log p(x_i) - log q(x_i)))
```

### Tiling strategy

The full logprob tensors have shape `(B, γ, V)` where V ≈ 32,000.
**We do NOT sweep the vocab dimension.**  The acceptance test at position `i`
requires only two scalars: `lp_q[b, i, token_id]` and `lp_p[b, i, token_id]`.

Grid: `(ceil(B / BLOCK_B), γ)`.  Each program:
- Reads `token_id[b, i]`  — 4 bytes per element.
- Reads `lp_q[b, i, token_id]` and `lp_p[b, i, token_id]`  — 8 bytes each.
- Writes `accepted[b, i]`  — 1 byte per element.

**SRAM footprint**: O(BLOCK_B) scalars — constant in V.

### Autotune configs

| BLOCK_B | Warps | Notes |
|---------|-------|-------|
| 1       | 1     | Minimal latency for B=1 |
| 2       | 2     | |
| 4       | 4     | Default balanced |
| 8       | 4     | High throughput for B≥8 |

---

## 2.2 Gather kernel (`gather_kernel.py`)

### Problem

Mask out draft token IDs at positions ≥ `first_rejection[b]`.

### Tiling strategy

Grid: `(ceil(B*γ / BLOCK_SIZE),)`.  Each program handles a contiguous
chunk of the flattened `(B*γ)` token sequence.

For each element at flat offset `k`:
- `batch_idx = k // γ`
- `gamma_pos = k % γ`
- Accept iff `gamma_pos < first_rejection[batch_idx]`, else write `-1`.

**SRAM footprint**: O(BLOCK_SIZE) integers — constant in V.

### Autotune configs

| BLOCK_SIZE | Warps | Notes |
|------------|-------|-------|
| 64         | 2     | Small batches |
| 128        | 4     | |
| 256        | 4     | Default |
| 512        | 8     | Maximum occupancy |

---

## Numerical precision

- All arithmetic performed in **float32** even when input is bfloat16.
- Tolerance contract: `max(|kernel - reference|) < 1e-5` (float32),
  `< 1e-3` (bfloat16).
- Verified by `tests/unit/test_verify_kernel.py`.

---

## Roofline analysis, not a profile

The figures below are arithmetic from the NVIDIA H100 whitepaper
(989 TFLOP/s bfloat16, 3.35 TB/s HBM). They were not produced by a
profiler. No H100 run has been committed, and this section does not
add one. The measured T4 comparison is in the README: at batch 1 the
Triton kernel was slower than the pure-PyTorch reference.

- The verification kernel is memory-bandwidth bound on paper (it reads 2 scalars per token).
- At 3.35 TB/s, a gather-limited estimate for B=32, γ=4 is about 6 μs.
- Avoiding a vocabulary sweep is about 14× fewer bytes than a naive scan. That ratio is a byte count, not a measured speedup.

---

## Measurements (moved from the README)

The engine does not call these kernels. A speculative step's time is the
drafter and target forwards (milliseconds); the accept/reject test and
the residual are microseconds and run in PyTorch on the device.

- **Triton `verify_tokens` / `gather_accepted`** (`flashspec.kernels`).
  `verify_tokens` tiles over `(batch × γ)` positions; each thread reads
  two scalars (`log p`, `log q` at the draft token index) and writes one
  bool and one int32, so SRAM use does not grow with the vocabulary. It
  matches the pure-PyTorch reference numerically. There is no GPU CI
  runner: `notebooks/colab_triton_tests.py` runs the `@pytest.mark.gpu`
  parity tests on Colab and writes `colab_triton_tests.log`.
- **Assembly residual** (`flashspec-asm-kernel`, the `[asm]` extra,
  Linux x86-64). A residual evaluation is microseconds against
  millisecond model forwards, so it does not change end-to-end
  throughput.

### Triton verification kernel vs the PyTorch reference — measured (Tesla T4)

Saved output of `notebooks/03_kernel_profiling.ipynb` cell 3 (0-based
index, execution count 4), printed there as a Tesla T4 run. At batch 1
the Triton kernel is slower than the PyTorch reference.

| Shape | Reference (ms) | Triton (ms) | Speedup |
|---|---|---|---|
| B=1 γ=4 V=32k | 0.151 | 0.785 | 0.2× |
| B=8 γ=4 V=32k | 0.717 | 0.459 | 1.6× |
| B=1 γ=8 V=32k | 0.231 | 1.248 | 0.2× |
| B=32 γ=4 V=32k | 0.091 | 0.222 | 0.4× |

> The Triton kernel is slower than the pure-PyTorch reference at batch 1
> on this T4 measurement. A faster H100 result has not been measured.
> Do not treat the H100 roofline above as a profile.

### Assembly residual (`[asm]` extra) — Linux, git `4e48c55`, then git `1b0cda3`

Milliseconds, backend `asm`. Compare backends inside one run. The two runs do not repeat the same ordering.

| Vocab | PyTorch residual | C baseline | ASM residual | PyTorch verify | ASM verify |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0118 ± 0.00599 | 0.0109 ± 0.00177 | 0.0151 ± 0.00971 | 0.366 ± 0.0209 | 0.422 ± 0.0438 |
| 4096 | 0.0116 ± 0.000285 | 0.0140 ± 0.00214 | 0.0140 ± 0.00430 | 0.495 ± 0.0218 | 0.537 ± 0.0159 |
| 32000 | 0.0302 ± 0.00818 | 0.0430 ± 0.00268 | 0.0248 ± 0.0230 | 1.28 ± 0.638 | 1.32 ± 0.658 |

| Vocab | PyTorch residual | C baseline | ASM residual | PyTorch verify | ASM verify |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.00686 ± 0.00312 | 0.00609 ± 0.000588 | 0.00806 ± 0.00108 | 0.158 ± 0.0421 | 0.164 ± 0.0305 |
| 4096 | 0.00685 ± 0.000109 | 0.00785 ± 0.000559 | 0.00872 ± 0.000941 | 0.231 ± 0.0339 | 0.250 ± 0.0390 |
| 32000 | 0.0124 ± 0.00209 | 0.0235 ± 0.00266 | 0.0151 ± 0.0187 | 0.747 ± 0.393 | 0.730 ± 0.376 |
