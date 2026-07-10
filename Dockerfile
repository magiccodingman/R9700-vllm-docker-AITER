# syntax=docker/dockerfile:1.7

ARG ROCM_BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.4-complete
FROM ${ROCM_BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2
ARG PYTORCH_PACKAGES="torch torchvision torchaudio"
ARG AMD_TRITON_INDEX_URL=https://pypi.amd.com/triton/release_/rocm-7.2.0/simple/
ARG TRITON_PACKAGES="triton==3.7.0 triton-kernels==1.0.0"
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=55d6e42f9b809f0c40b23562525fe7354622b085
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_BASE_COMMIT=735def4fcf39945b6e6c24769878760e3e113b15
ARG VLLM_FINAL_BRANCH=r9700-c3284-secondary
ARG LAUNCHER_REPO=https://github.com/magiccodingman/VllmLaunchScriptR9700.git
ARG LAUNCHER_REF=main
ARG MAX_JOBS=8
ARG PYTORCH_ROCM_ARCH=gfx1201
ARG PIP_DEFAULT_TIMEOUT=7200
ARG PIP_RETRIES=100

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
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT} \
    PIP_RETRIES=${PIP_RETRIES} \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
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

RUN cat > /usr/local/bin/check-rocm-torch <<'PY'
#!/usr/bin/env python3
import torch

hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
print("torch:", torch.__version__)
print("hip:", hip)
print("cuda:", cuda)
print("torch file:", torch.__file__)
assert hip is not None, "BROKEN: torch is not ROCm/HIP-enabled"
assert cuda is None, "BROKEN: CUDA torch replaced the ROCm torch stack"
PY
RUN chmod +x /usr/local/bin/check-rocm-torch

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
      ${PYTORCH_PACKAGES} \
    && check-rocm-torch

RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages --force-reinstall --ignore-installed \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      --extra-index-url "${AMD_TRITON_INDEX_URL}" \
      ${TRITON_PACKAGES} \
    && python -m pip install --break-system-packages --ignore-installed \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "numpy==2.1.3" \
    && check-rocm-torch

# vLLM is installed with --no-build-isolation so the build backend dependencies
# must already exist in the system Python environment.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "setuptools-rust" \
    && check-rocm-torch

# Build/install AITER from the exact commit used in the manual process.
# AITER's setup.py may shell out to `python -m pip install flydsl==...`.
# PIP_BREAK_SYSTEM_PACKAGES=1 above lets those internal pip subprocesses work
# in this sealed Docker image without patching upstream setup.py.
#
# Do not let `setup.py develop` use old easy_install dependency processing.
# It can treat binary console scripts from wheel eggs as UTF-8 metadata and
# explode on packages such as ninja. Preinstall the deps with pip, then run
# develop with --no-deps.
#
# Do not `import aiter` during docker build. Importing AITER can trigger JIT/GPU
# arch detection through rocminfo, and docker build does not have the runtime
# /dev/kfd and /dev/dri device wiring from Compose.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    cd /opt/r9700-vllm/src \
    && rm -rf aiter \
    && git clone --recursive "${AITER_REPO}" aiter \
    && cd aiter \
    && git checkout "${AITER_COMMIT}" \
    && git submodule sync \
    && git submodule update --init --recursive \
    && git rev-parse HEAD | tee /opt/r9700-vllm/build-info/aiter.commit.txt \
    && python -m pip install --break-system-packages \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "flydsl==0.2.2" \
      "einops" \
      "pandas" \
      "pybind11>=3.0.1" \
      "python-dateutil>=2.8.2" \
      "six>=1.5" \
      "psutil" \
      "ninja" \
    && unset SCCACHE_BUCKET SCCACHE_REGION SCCACHE_ENDPOINT SCCACHE_S3_USE_SSL SCCACHE_S3_KEY_PREFIX \
             SCCACHE_IDLE_TIMEOUT SCCACHE_ERROR_LOG SCCACHE_LOG RUSTC_WRAPPER \
             CUDA_HOME CUDA_PATH CUDA_ROOT CUDA_VISIBLE_DEVICES TORCH_CUDA_ARCH_LIST NVCC_PREPEND_FLAGS \
    && export CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc CMAKE_CXX_COMPILER=/usr/bin/g++ \
    && python3 setup.py develop --no-deps 2>&1 | tee /opt/r9700-vllm/build-info/aiter-build.log \
    && python - <<'PY'
import importlib.metadata as md
import importlib.util
print('AITER distribution:', md.version('amd-aiter'))
spec = importlib.util.find_spec('aiter')
assert spec is not None, 'aiter module spec not found'
print('AITER module spec:', spec.origin)
PY
RUN check-rocm-torch

# Clone vLLM, apply the pinned RDNA4/R9700 patch stack, then build/install it for ROCm.
# Build vLLM itself with --no-deps so pip does not replace ROCm torch/triton
# with PyPI CUDA/NVIDIA wheels.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    cd /opt/r9700-vllm/src \
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
    && export VLLM_TARGET_DEVICE=rocm MAX_JOBS="${MAX_JOBS}" PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
    && python -m pip install --break-system-packages --no-build-isolation --no-deps -v -e . 2>&1 | tee /opt/r9700-vllm/build-info/vllm-build.log \
    && check-rocm-torch

# After the wheel is installed, read vLLM's own package metadata, filter only
# CUDA/NVIDIA-sensitive package names, and install the remaining runtime
# dependencies under a constraints file that pins the ROCm torch/triton/numpy stack.
RUN python - <<'PY'
import importlib.metadata as md
from pathlib import Path

pins = ['tokenizers==0.22.2']
for name in ('torch', 'torchvision', 'torchaudio', 'triton', 'triton-kernels', 'numpy'):
    try:
        pins.append(f'{name}=={md.version(name)}')
    except md.PackageNotFoundError:
        pass
Path('/tmp/rocm-python-constraints.txt').write_text('\n'.join(pins) + '\n')
print('ROCm Python constraints:')
print('\n'.join(pins))
PY

RUN python - <<'PY'
import importlib.metadata as md
from packaging.requirements import Requirement
from pathlib import Path

skip_exact = {
    'torch',
    'torchvision',
    'torchaudio',
    'triton',
    'triton-kernels',
    'cuda-toolkit',
    'cuda-bindings',
    'cuda-pathfinder',
}
skip_prefixes = ('nvidia-', 'cuda-')
requirements = []
for raw in md.distribution('vllm').requires or []:
    req = Requirement(raw)
    name = req.name.lower().replace('_', '-')
    if name in skip_exact or any(name.startswith(prefix) for prefix in skip_prefixes):
        print(f'Skipping ROCm/CUDA-sensitive dependency: {raw}')
        continue
    if req.marker is not None and not req.marker.evaluate({'extra': ''}):
        continue
    requirements.append(raw)

Path('/tmp/vllm-runtime-requirements.txt').write_text('\n'.join(requirements) + '\n')
print('vLLM runtime dependency count:', len(requirements))
for req in requirements:
    print(req)
PY

RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages --ignore-installed --no-deps \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      --extra-index-url "${PYTORCH_INDEX_URL}" \
      --constraint /tmp/rocm-python-constraints.txt \
      -r /tmp/vllm-runtime-requirements.txt \
    && check-rocm-torch

# The direct vLLM install above intentionally uses --no-deps to protect the
# ROCm torch/triton stack. Hydrate known non-GPU transitive runtime deps
# explicitly, without -U, and keep the ROCm guard active immediately after.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages --ignore-installed \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      --constraint /tmp/rocm-python-constraints.txt \
      uvloop urllib3 \
      certifi charset_normalizer idna \
      aiohappyeyeballs aiosignal attrs frozenlist multidict propcache yarl \
      anyio h11 httpcore click rich shellingham python-dotenv \
      annotated-doc annotated-types pydantic-core pydantic-extra-types pycountry \
      cffi cryptography pycparser \
      filelock fsspec pyyaml \
      jsonschema-specifications referencing rpds-py \
      dill httpx huggingface-hub multiprocess pyarrow xxhash \
      distro docstring-parser jiter sniffio astor jmespath supervisor \
      httpx-sse pydantic-settings "pyjwt[crypto]" python-multipart \
      sse-starlette typing-inspection uvicorn typer \
    && check-rocm-torch

# peft may need accelerate at runtime, but installing accelerate normally after
# ROCm torch is present can make pip try to replace torch. Install it isolated.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages --ignore-installed --no-deps \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      accelerate \
    && check-rocm-torch

RUN python - <<'PY'
import aiohttp
import anyio
import httpx
import jsonschema
import pycountry
import referencing
import tokenizers
import transformers
from jsonschema import Draft7Validator
from multidict import istr
from pydantic_extra_types.language_code import LanguageAlpha2
from transformers import MistralCommonBackend, PretrainedConfig
from vllm.connections import global_http_connection
print('aiohttp:', aiohttp.__version__)
print('jsonschema:', jsonschema.__version__)
print('pycountry import: ok', pycountry.__version__)
print('tokenizers:', tokenizers.__version__)
print('transformers:', transformers.__version__)
print('pydantic extra LanguageAlpha2 import: ok', LanguageAlpha2)
print('vLLM connection import: ok', type(global_http_connection).__name__)
print('transformers MistralCommonBackend import: ok', MistralCommonBackend)
print('transformers PretrainedConfig import: ok', PretrainedConfig)
PY
RUN check-rocm-torch

RUN python - <<'PY'
import vllm
print('vLLM loaded from:', vllm.__file__)
PY
RUN check-rocm-torch

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
