# FlashSpec server image: the released PyPI package on a CUDA 13.0 base.
#
#   docker build -f deploy/Dockerfile --build-arg FLASHSPEC_VERSION=0.3.0 -t flashspec deploy/
#   docker run --gpus all -p 8000:8000 flashspec
#
# GPU-oriented: torch is the CUDA 13.0 build FlashSpec's GPU results were
# measured with (torch 2.11.0+cu130, transformers 5.20.2, Tesla T4). It also
# runs on CPU (pass --device cpu --dtype float32), but CPU is not where
# FlashSpec is faster. The default command serves the measured pair,
# Qwen2.5-1.5B-Instruct verified by Qwen2.5-0.5B-Instruct, in fp16; any
# arguments you pass replace it (see `python -m flashspec.serve --help`).
#
# Built and pushed to ghcr.io/mattral/flashspec by .github/workflows/container.yml.

# The "base" flavour is the smallest CUDA image; torch's wheel brings its own
# CUDA libraries. Pinned by digest (linux/amd64) for reproducible builds.
FROM nvidia/cuda:13.0.0-base-ubuntu24.04@sha256:cf5ab24c3f5040a0ea5931658874c2158e606bf740f006809aa4bb9d3334016b

ARG FLASHSPEC_VERSION
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TRANSFORMERS_VERSION=5.20.2

# Our own labels (they also replace the base image's maintainer and ref.name).
LABEL maintainer="Min Htet Myet (Mattral)" \
      org.opencontainers.image.authors="Min Htet Myet (Mattral)" \
      org.opencontainers.image.ref.name="flashspec" \
      org.opencontainers.image.base.name="docker.io/nvidia/cuda:13.0.0-base-ubuntu24.04" \
      org.opencontainers.image.source="https://github.com/Mattral/FlashSpec" \
      org.opencontainers.image.title="flashspec" \
      org.opencontainers.image.description="FlashSpec: lossless speculative decoding for HuggingFace causal LMs, served over an OpenAI-compatible HTTP API (GPU-oriented, CUDA 13.0)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${FLASHSPEC_VERSION}" \
      org.opencontainers.image.url="https://github.com/Mattral/FlashSpec"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Python 3.12 from Ubuntu 24.04, in a venv (PEP 668 blocks system-wide pip).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# torch from the CUDA 13.0 index first, so pip keeps it when it resolves flashspec.
RUN test -n "${FLASHSPEC_VERSION}" || (echo "--build-arg FLASHSPEC_VERSION is required" && exit 1) \
    && pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && pip install "flashspec[serve]==${FLASHSPEC_VERSION}" "transformers==${TRANSFORMERS_VERSION}" \
    && python -c "import flashspec, torch; assert flashspec.__version__ == '${FLASHSPEC_VERSION}', flashspec.__version__; print('flashspec', flashspec.__version__, 'torch', torch.__version__)"

# Non-root user; downloaded checkpoints go to its HuggingFace cache (mount a
# volume at /home/flashspec/.cache/huggingface to keep them between runs).
# (UID 1000 is taken by the base image's `ubuntu` user.)
RUN useradd --create-home --uid 10001 flashspec
USER flashspec
WORKDIR /home/flashspec
ENV HF_HOME=/home/flashspec/.cache/huggingface

EXPOSE 8000
ENTRYPOINT ["python", "-m", "flashspec.serve"]
CMD ["--target", "Qwen/Qwen2.5-1.5B-Instruct", "--drafter", "Qwen/Qwen2.5-0.5B-Instruct", \
     "--device", "auto", "--dtype", "float16", "--host", "0.0.0.0", "--port", "8000"]
