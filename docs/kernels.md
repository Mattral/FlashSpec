# FlashSpec Triton Kernels

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

## Roofline analysis

The figures below are arithmetic from the NVIDIA H100 whitepaper
(989 TFLOP/s bfloat16, 3.35 TB/s HBM). They are not a profile of this
repository. No H100 run has been committed. The measured T4 comparison
is in the README: at batch 1 the Triton kernel was slower than the
pure-PyTorch reference.

- The verification kernel is memory-bandwidth bound on paper (it reads 2 scalars per token).
- At 3.35 TB/s, a gather-limited estimate for B=32, γ=4 is about 6 μs.
- Avoiding a vocabulary sweep is about 14× fewer bytes than a naive scan. That ratio is a byte count, not a measured speedup.
