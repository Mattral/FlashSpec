# Postponed

These items are waiting on a package upload or a GPU. They are not
blockers. CPU work, the test suite, and the single-device server can
move without them.

## Publishing

- Releases go to PyPI through `publish.yml` (trusted publishing, run by
  hand on a `v<version>` tag; see `PUBLISHING.md`). 0.2.0 is on PyPI with a
  provenance attestation for both files, and its GitHub Release is
  published.
- The Zenodo record 10.5281/zenodo.20766119 was still the 2026-06-19
  version when checked on 2026-10-26. The draft for a later deposit is
  `paper/flashspec.tex` (October 2026, software 0.2.0). Updating and
  depositing it stays with the maintainer.
- The PyPI project `flashspec` at 0.1.2 was yanked by the maintainer.
  Do not install `flashspec==0.1.2`. There is no `v0.1.2` tag.
  `flashspec-asm-kernel` 0.1.2 is a different package.

## Packaging

- `flashspec-asm-kernel` 0.1.2 on PyPI is a manylinux x86-64 wheel and
  contains `libflashspec_asm.so`. The `asm` extra requires >= 0.1.2.
  Release 0.1.0 was an sdist with no shared library. This checkout has
  not loaded the 0.1.2 wheel. The recorded files
  `benchmarks/results/residual_linux_gha.json` and
  `benchmarks/results/residual_linux_gha_1b0cda3.json` are source builds.
- `notebooks/04_gpu_measurement.ipynb` still clones flashspec-asm and
  runs `make` when the wheel does not select the kernel.

## Generation measurements

- A longer CPU run than the one HumanEval/23 sample at git `f59b5b4`
  (`n_samples=1`, 16 new tokens, `n_warmup=0`).
- A longer TinyLlama T4 run than the recorded `n_samples=8` pair.
- Two model drafters on a real target. Phase 6 (bundle `3e38251`) measured
  one model drafter (Qwen2.5-0.5B) and prompt lookup as bandit arms; two
  different model drafters have not been measured.
- A larger target with the same drafter (Qwen2.5-3B-Instruct on the T4;
  no config exists yet; watch for NaN/inf in fp16).
- Batch sizes above 1 on a GPU. Only batch 1 has been measured.
- The contextual bandit recorded as future work in the README. It is not
  implemented.
- Llama-3-8B, or any larger target, is not scheduled, and neither is an
  H100 run (it costs money). 0.2.0 and 0.3.0 were measured on one Tesla T4
  only.

This repository has no self-hosted GPU runner, so it has no GPU
workflow. The former GPU Tests and H100 benchmark workflows were removed.
GPU measurements run by hand on a Colab T4 with
`notebooks/04_gpu_measurement.ipynb`.
