# Postponed

These items are waiting on a package upload or a GPU. They are not
blockers. CPU work, the test suite, and the single-device server can
move without them.

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

- A longer TinyLlama T4 run than the recorded `n_samples=8` pair.
- Two real drafters on a real model. The synthetic two-drafter switch
  is already measured. The T4 pair used one drafter.
- Llama-3-8B, or any larger target. That needs about 24 GB and compute
  capability 8 or newer. v0.2.0 stays reserved for a measured
  datacenter-GPU Llama run. Do not tag 0.2.0 before that measurement
  exists.

This repository has no self-hosted GPU runner. GPU Tests and the H100
benchmark start only from `workflow_dispatch` after a runner exists.

## Local checks

- mypy on the Windows workstation. Application Control blocks the mypyc
  extension there, so that check is postponed on that machine.
