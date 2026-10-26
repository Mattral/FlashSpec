# Installation

The README has the one-line install and the extras table. This page has
the rest: every install route, platform notes, and the requirements.

## From PyPI, from source, or with Docker

```bash
# PyPI — cross-platform (Windows/macOS/Linux):
pip install flashspec

# PyPI — adds Triton (Linux only): the experimental Triton kernels.
# Linux CUDA builds of PyTorch already ship Triton, which torch.compile
# needs for the compiled path.
pip install flashspec[gpu]

# Optional AVX2 residual kernel. The wheel is a Linux x86-64 ELF library.
# On Windows and macOS the extra installs nothing and the PyTorch residual stays selected.
pip install flashspec[asm]

# Optional single-device HTTP API (FastAPI). One GPU or CPU, no streaming.
pip install flashspec[serve]

# From source:
git clone https://github.com/Mattral/FlashSpec && cd FlashSpec
pip install -e ".[dev,gpu]"   # omit ",gpu" on Windows/macOS

# Container image (Linux + NVIDIA GPU), published for each release:
docker pull ghcr.io/mattral/flashspec:0.3.0
docker run --gpus all -p 8000:8000 ghcr.io/mattral/flashspec:0.3.0
```

**The container image** (`ghcr.io/mattral/flashspec:<version>` and `:latest`) is
built by `.github/workflows/container.yml` when a release is published:

- It installs `flashspec[serve]==<version>` from PyPI on `nvidia/cuda:13.0.0-base-ubuntu24.04`,
  with torch `2.11.0+cu130` and transformers `5.20.2`, the build the GPU results were measured
  with.
- It runs as the non-root user `flashspec`; downloaded checkpoints go to
  `/home/flashspec/.cache/huggingface`, so mount a volume there to keep them.
- Its entrypoint is `python -m flashspec.serve`. The default arguments serve
  Qwen2.5-1.5B-Instruct verified by Qwen2.5-0.5B-Instruct in fp16 on port 8000. Arguments you
  pass replace them, so include `--host 0.0.0.0`.
- The workflow smoke-tests every image on a CPU runner (import, then the server with a tiny
  model) before it pushes. `deploy/docker-compose.yml` and
  `deploy/k8s/flashspec-deployment.yaml` use the published image.

To build it yourself from a released version:

```bash
docker build -f deploy/Dockerfile --build-arg FLASHSPEC_VERSION=0.3.0 -t flashspec deploy/
```

A development checkout, by platform:

```bash
git clone https://github.com/Mattral/FlashSpec && cd FlashSpec

# Linux + CUDA (accelerated Triton kernels):
pip install -e ".[dev,gpu]"

# Windows / macOS (pure-PyTorch reference kernels, numerically identical):
pip install -e ".[dev]"

# Optional CPU residual kernel (Linux x86-64, flashspec-asm-kernel>=0.1.2;
# other platforms install nothing):
pip install -e ".[asm]"

# Optional single-device HTTP server:
pip install -e ".[serve]"
```

**`accelerate`.** `python -m flashspec.serve` loads the target with
`device_map`, which transformers only accepts when `accelerate` is
installed, so the `serve` extra includes `accelerate` (0.2.0's did not).
The same applies to `TargetModel(model_id, device)` without `model=` in
your own code: install `accelerate`, or load the module yourself and pass
it with `model=`, as the README quickstart does.

> **Windows / macOS:** `pip install flashspec` works out of the box.
> Triton has no official PyPI wheels outside Linux, so the compiled path
> and the experimental Triton kernels are Linux + CUDA only. The engine's
> accept/reject test is PyTorch and runs everywhere PyTorch does. The
> kernels' pure-PyTorch reference is `flashspec.kernels._reference`.
> Check availability at runtime: `from flashspec import TRITON_AVAILABLE`.

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | |
| PyTorch | ≥ 2.2 | CPU or CUDA |
| Triton | ≥ 3.0 | Optional · Linux + CUDA · needed by the compiled path (Linux CUDA PyTorch ships it) and the experimental kernels |
| flashspec-asm-kernel | ≥ 0.1.2, < 2 | Optional · Linux x86-64 · `pip install flashspec[asm]`. 0.1.2 ships a manylinux wheel with `libflashspec_asm.so`. |
| FastAPI, uvicorn, httpx, accelerate | see `serve` extra | Optional · `pip install flashspec[serve]` |
| CUDA | ≥ 12.0 | GPU path only |
| transformers | ≥ 4.40 | For HuggingFace model loading |

## The `asm` extra

The `asm` extra installs `flashspec-asm-kernel` >= 0.1.2 and < 2 on Linux
x86-64. Release 0.1.2 is a manylinux wheel that contains
`libflashspec_asm.so`. Release 0.1.0 was an sdist with no shared library.
An import error, or an `OSError` from `abi_version()`, keeps the
pure-PyTorch residual. A residual evaluation is microseconds, and a model
forward is milliseconds, so this kernel does not change end-to-end
throughput. `benchmarks/results/residual_linux_gha.json` and
`benchmarks/results/residual_linux_gha_1b0cda3.json` are source builds
from the flashspec-asm repo, not a timing of the 0.1.2 wheel.
The PyPI project `flashspec` at version 0.1.2 was yanked by the maintainer.
Do not install `flashspec==0.1.2`. That yank does not apply to
`flashspec-asm-kernel`.
