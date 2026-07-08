# syntax=docker/dockerfile:1.7

ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4-complete
FROM ${ROCM_BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2
ARG PYTORCH_PACKAGES="torch torchvision torchaudio"
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=55d6e42f9b809f0c40b23562525fe7354622b085
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_BASE_COMMIT=c3284c31f52c005bde02cf7899959c2539b01d2f
ARG VLLM_FINAL_BRANCH=r9700-c3284-secondary
ARG LAUNCHER_REPO=https://github.com/magiccodingman/VllmLaunchScriptR9700.git
ARG LAUNCHER_REF=main
ARG MAX_JOBS=8
ARG PIP_DEFAULT_TIMEOUT=1200
ARG PIP_RETRIES=20

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH} \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TORCH_HOME=/cache/torch \
    TRITON_CACHE_DIR=/cache/triton \
    VLLM_CACHE_ROOT=/cache/vllm \
    PIP_CACHE_DIR=/cache/pip \
    PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT} \
    PIP_RETRIES=${PIP_RETRIES} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_WORKSPACE=/opt/r9700-vllm

WORKDIR /opt/r9700-vllm

COPY docker/apply-vllm-stack.sh /usr/local/bin/apply-vllm-stack
COPY docker/r9700-entrypoint.sh /usr/local/bin/r9700-entrypoint
COPY docker/verify-rocm.py /usr/local/bin/verify-rocm.py

RUN chmod +x /usr/local/bin/apply-vllm-stack /usr/local/bin/r9700-entrypoint /usr/local/bin/verify-rocm.py

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs curl wget ca-certificates \
      build-essential cmake ninja-build pkg-config \
      python3 python3-dev python3-pip python3-setuptools python3-wheel python3-packaging python3-setuptools-scm python-is-python3 \
      rustc cargo \
      numactl libnuma-dev \
      jq less vim-tiny procps tini \
    && git lfs install --system \
    && mkdir -p /opt/r9700-vllm/src /opt/r9700-vllm/build-info \
                /cache/pip /cache/vllm /cache/torch /cache/triton /cache/huggingface /logs \
    && rm -rf /var/lib/apt/lists/*

# System Python only. No venv. Do not upgrade apt-owned Python build tools
# such as pip/setuptools/wheel/packaging with pip; Debian packages often lack
# wheel RECORD metadata, so pip cannot uninstall them cleanly.
#
# PyTorch ROCm wheels are huge, so this uses long pip timeouts/retries and a
# BuildKit cache mount. Once the torch layer succeeds, later build failures do
# not force another multi-GB torch download.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages \
      --index-url "${PYTORCH_INDEX_URL}" \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      ${PYTORCH_PACKAGES}

RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages --force-reinstall --ignore-installed \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      --extra-index-url https://pypi.amd.com/triton/release_/rocm-7.2.0/simple/ \
      "triton==3.7.0" \
      "triton-kernels==1.0.0" \
    && python -m pip install --break-system-packages --ignore-installed \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "numpy==2.1.3" \
    && python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch file:', torch.__file__)
print('HIP:', torch.version.hip)
PY

# Build/install AITER from the exact commit used in the manual process.
RUN cd /opt/r9700-vllm/src \
    && rm -rf aiter \
    && git clone --recursive "${AITER_REPO}" aiter \
    && cd aiter \
    && git checkout "${AITER_COMMIT}" \
    && git submodule sync \
    && git submodule update --init --recursive \
    && git rev-parse HEAD | tee /opt/r9700-vllm/build-info/aiter.commit.txt \
    && unset SCCACHE_BUCKET SCCACHE_REGION SCCACHE_ENDPOINT SCCACHE_S3_USE_SSL SCCACHE_S3_KEY_PREFIX \
             SCCACHE_IDLE_TIMEOUT SCCACHE_ERROR_LOG SCCACHE_LOG RUSTC_WRAPPER \
             CUDA_HOME CUDA_PATH CUDA_ROOT CUDA_VISIBLE_DEVICES TORCH_CUDA_ARCH_LIST NVCC_PREPEND_FLAGS \
    && export CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc CMAKE_CXX_COMPILER=/usr/bin/g++ \
    && python3 setup.py develop 2>&1 | tee /opt/r9700-vllm/build-info/aiter-build.log \
    && python - <<'PY'
import aiter
print('AITER loaded from:', aiter.__file__)
PY

# Clone vLLM, apply the pinned RDNA4/R9700 patch stack, then build/install it for ROCm.
RUN cd /opt/r9700-vllm/src \
    && rm -rf vllm \
    && git clone "${VLLM_REPO}" vllm \
    && cd vllm \
    && git remote rename origin upstream \
    && git remote add ggz14 https://github.com/GGZ14/vllm.git \
    && git remote add ar https://github.com/A-R-Dhedeep-Reddy/vllm.git \
    && git remote add feiyehua https://github.com/feiyehua/vllm.git \
    && VLLM_BASE_COMMIT="${VLLM_BASE_COMMIT}" apply-vllm-stack \
    && git checkout "${VLLM_FINAL_BRANCH}" \
    && git rev-parse HEAD | tee /opt/r9700-vllm/build-info/vllm.final.commit.txt \
    && unset SCCACHE_BUCKET SCCACHE_REGION SCCACHE_ENDPOINT SCCACHE_S3_USE_SSL SCCACHE_S3_KEY_PREFIX \
             SCCACHE_IDLE_TIMEOUT SCCACHE_ERROR_LOG SCCACHE_LOG RUSTC_WRAPPER \
             CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER \
             CUDA_HOME CUDA_PATH CUDA_ROOT CUDA_VISIBLE_DEVICES TORCH_CUDA_ARCH_LIST NVCC_PREPEND_FLAGS \
    && export CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc CMAKE_CXX_COMPILER=/usr/bin/g++ \
    && export VLLM_TARGET_DEVICE=rocm MAX_JOBS="${MAX_JOBS}" \
    && python -m pip install --break-system-packages --no-build-isolation -v -e . 2>&1 | tee /opt/r9700-vllm/build-info/vllm-build.log \
    && python - <<'PY'
import vllm
print('vLLM loaded from:', vllm.__file__)
PY

# Pull the launch wrapper, but keep it runtime-overridable with VLLM_LAUNCH_SCRIPT.
RUN cd /opt/r9700-vllm \
    && rm -rf launcher \
    && git clone "${LAUNCHER_REPO}" launcher \
    && cd launcher \
    && git checkout "${LAUNCHER_REF}" \
    && git rev-parse HEAD | tee /opt/r9700-vllm/build-info/launcher.commit.txt

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/r9700-entrypoint"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
