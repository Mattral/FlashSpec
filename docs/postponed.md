# Postponed

These items are waiting on a package upload or a GPU. They are not
blockers. CPU work, the test suite, and the single-device server can
move without them.

## Publishing

- 0.1.5 is on PyPI as a wheel and an sdist, and it has a GitHub Release.
  The Zenodo record 10.5281/zenodo.20766119 was still the 2026-06-19
  version when last checked. The draft for a later deposit is
  `paper/flashspec.tex` (October 2026, software 0.2.0). Publishing that
  version stays with the maintainer.
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
- Two real drafters on a real model. The synthetic two-drafter switch
  is already measured. The T4 pair used one drafter.
- Llama-3-8B, or any larger target, is not scheduled. The v0.2.0 H100
  run is skipped unless a specific reason comes up. It costs money. 0.2.0
  was measured on one Tesla T4 only.

This repository has no self-hosted GPU runner, so it has no GPU
workflow. The former GPU Tests and H100 benchmark workflows were removed.
GPU measurements run by hand on a Colab T4 with
`notebooks/04_gpu_measurement.ipynb`.
