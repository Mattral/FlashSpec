# FlashSpec production Docker image
# Base: NVIDIA CUDA 12.4 + Python 3.11
# Triton requires a CUDA-capable GPU at runtime.

ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

LABEL maintainer="Min Htet Myet (Mattral)"
LABEL description="FlashSpec adaptive speculative-decoding inference engine"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /workspace

COPY pyproject.toml ./
# Install PyTorch separately to pick the right CUDA wheel.
RUN pip install --no-cache-dir \
        torch==2.3.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124

# Install the package in editable mode.
COPY . .
RUN pip install --no-cache-dir -e ".[onnx]"

# ── Runtime ───────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV TRITON_CACHE_DIR=/tmp/triton_cache

# Smoke test during build.
RUN python -c "import flashspec; print('FlashSpec', flashspec.__version__, 'loaded OK')"

CMD ["python", "-c", "import flashspec; print('FlashSpec ready.')"]
