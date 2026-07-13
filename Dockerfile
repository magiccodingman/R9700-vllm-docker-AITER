# syntax=docker/dockerfile:1.7

ARG ROCM_BASE_IMAGE=ubuntu:24.04
FROM ${ROCM_BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCM_VERSION=7.13.0
ARG ROCM_TARBALL_URL=https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx120X-all-7.13.0.tar.gz
ARG PYTORCH_INDEX_URL=https://repo.amd.com/rocm/whl/gfx120X-all/
ARG PYTORCH_PACKAGES="torch==2.11.0+rocm7.13.0 torchvision==0.26.0+rocm7.13.0 torchaudio==2.11.0+rocm7.13.0"
ARG TRITON_PACKAGE=triton==3.6.0+rocm7.13.0
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_COMMIT=55d6e42f9b809f0c40b23562525fe7354622b085
ARG VLLM_REPO=https://github.com/magiccodingman/vllm-rdna4.git
ARG VLLM_REF=rdna4-dev
ARG MAX_JOBS=8
ARG PYTORCH_ROCM_ARCH=gfx1201
ARG PIP_DEFAULT_TIMEOUT=7200
ARG PIP_RETRIES=100

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH} \
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

COPY docker/r9700-entrypoint.sh /usr/local/bin/r9700-entrypoint
COPY docker/verify-rocm.py /usr/local/bin/verify-rocm.py

RUN chmod +x /usr/local/bin/r9700-entrypoint /usr/local/bin/verify-rocm.py

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs curl wget ca-certificates \
      build-essential cmake ninja-build pkg-config \
      python3 python3-dev python3-pip python3-setuptools python3-wheel python3-packaging python3-setuptools-scm python-is-python3 \
      rustc cargo \
      numactl libnuma-dev libatomic1 libquadmath0 libgfortran5 \
      jq less vim-tiny procps tini \
    && git lfs install --system \
    && mkdir -p /opt/r9700-vllm/src /opt/r9700-vllm/build-info \
                /cache/pip /cache/vllm /cache/torch /cache/triton /cache/huggingface /logs \
    && rm -rf /var/lib/apt/lists/*

# ROCm 7.13.0 Technology Preview is distributed through TheRock. Install the
# official gfx120X tarball as the complete userspace/development stack under
# /opt/rocm so AITER and vLLM build against the same ROCm version as PyTorch.
RUN wget --progress=dot:giga -O /tmp/rocm-${ROCM_VERSION}.tar.gz "${ROCM_TARBALL_URL}" \
    && mkdir -p "${ROCM_PATH}" \
    && tar -xf /tmp/rocm-${ROCM_VERSION}.tar.gz -C "${ROCM_PATH}" \
    && rm -f /tmp/rocm-${ROCM_VERSION}.tar.gz \
    && printf '%s\n' "${ROCM_VERSION}" | tee /opt/r9700-vllm/build-info/rocm.version.txt \
    && test -x "${ROCM_PATH}/bin/hipcc" \
    && "${ROCM_PATH}/bin/hipcc" --version

RUN cat > /usr/local/bin/check-rocm-torch <<'PY'
#!/usr/bin/env python3
import importlib.metadata as md
import torch

expected_torch = "2.11.0+rocm7.13.0"
expected_rocm = "7.13.0"
expected_hip_release = "7.13"
expected_triton = "3.6.0+rocm7.13.0"
rocm = md.version("rocm")
triton = md.version("triton")
try:
    triton_kernels = md.version("triton-kernels")
except md.PackageNotFoundError:
    triton_kernels = None
hip = getattr(torch.version, "hip", None)
hip_release = ".".join(hip.split(".")[:2]) if hip else None
cuda = getattr(torch.version, "cuda", None)
print("torch:", torch.__version__)
print("rocm package:", rocm)
print("hip:", hip)
print("cuda:", cuda)
print("triton:", triton)
print("triton-kernels:", triton_kernels)
print("torch file:", torch.__file__)
assert torch.__version__ == expected_torch, (
    f"BROKEN: expected torch {expected_torch}, got {torch.__version__}"
)
assert rocm == expected_rocm, (
    f"BROKEN: expected ROCm package {expected_rocm}, got {rocm}"
)
assert hip_release == expected_hip_release, (
    f"BROKEN: expected HIP release {expected_hip_release}.x, got {hip}"
)
assert triton == expected_triton, (
    f"BROKEN: expected Triton {expected_triton}, got {triton}"
)
assert triton_kernels is None, (
    f"BROKEN: unexpected triton-kernels package {triton_kernels}"
)
assert cuda is None, "BROKEN: CUDA torch replaced the ROCm torch stack"
PY
RUN chmod +x /usr/local/bin/check-rocm-torch

# Use AMD's gfx120X-all ROCm 7.13.0 wheel repository and exact Technology
# Preview package versions. --ignore-installed prevents pip from trying to
# uninstall apt-owned Python packages that do not have pip RECORD metadata.
RUN python -m pip install \
      --break-system-packages \
      --force-reinstall \
      --ignore-installed \
      --no-cache-dir \
      --index-url "${PYTORCH_INDEX_URL}" \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      ${PYTORCH_PACKAGES} \
    && check-rocm-torch

# PyTorch installs its matching ROCm 7.13 Triton dependency from the same AMD
# gfx120X index. Do not replace it with a Triton wheel from an older ROCm repo.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages --ignore-installed \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "numpy==2.1.3" \
    && check-rocm-torch

RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "setuptools-rust" \
    && check-rocm-torch

RUN python -m pip install --no-cache-dir loguru

# Build/install AITER from the selected commit.
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

# flydsl/AITER dependency resolution can pull AMD's older ROCm 7.2 Triton pair
# from PyPI. Remove that contamination and restore the official ROCm 7.13 Triton
# wheel from the gfx120X index before validating or building vLLM.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip uninstall -y triton triton-kernels pytorch-triton-rocm || true \
    && python -m pip install --break-system-packages --ignore-installed --no-deps \
      --index-url "${PYTORCH_INDEX_URL}" \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "${TRITON_PACKAGE}" \
    && check-rocm-torch

# The ROCm SMI CMake package pulled in by PyTorch requires libdrm's pkg-config
# metadata. Install the development package here, after AITER, so the expensive
# successful AITER build remains cached while vLLM gets the system dependency it
# needs for CMake configuration.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libdrm-dev \
    && pkg-config --modversion libdrm \
    && rm -rf /var/lib/apt/lists/*

# PyTorch's ROCm wheels install modern setuptools. Overlay a matching packaging
# release in /usr/local so setuptools does not fall back to Ubuntu's older
# apt-owned packaging module during vLLM editable metadata generation.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    python -m pip install --break-system-packages --ignore-installed \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --retries "${PIP_RETRIES}" \
      "packaging>=24.2" \
    && python -c "import packaging, packaging.licenses; print('packaging:', packaging.__version__)" \
    && check-rocm-torch

# Build vLLM directly from the selected ref in the RDNA4 fork. VLLM_REF defaults
# to rdna4-dev and may be overridden with any branch, tag, or commit.
RUN --mount=type=cache,target=/cache/pip,sharing=locked \
    cd /opt/r9700-vllm/src \
    && rm -rf vllm \
    && git clone "${VLLM_REPO}" vllm \
    && cd vllm \
    && git checkout "${VLLM_REF}" \
    && git rev-parse HEAD | tee /opt/r9700-vllm/build-info/vllm.commit.txt \
    && printf '%s\n' "${VLLM_REF}" | tee /opt/r9700-vllm/build-info/vllm.ref.txt \
    && unset SCCACHE_BUCKET SCCACHE_REGION SCCACHE_ENDPOINT SCCACHE_S3_USE_SSL SCCACHE_S3_KEY_PREFIX \
             SCCACHE_IDLE_TIMEOUT SCCACHE_ERROR_LOG SCCACHE_LOG RUSTC_WRAPPER \
             CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER \
             CUDA_HOME CUDA_PATH CUDA_ROOT CUDA_VISIBLE_DEVICES TORCH_CUDA_ARCH_LIST NVCC_PREPEND_FLAGS \
    && export CC=/usr/bin/gcc CXX=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc CMAKE_CXX_COMPILER=/usr/bin/g++ \
    && export VLLM_TARGET_DEVICE=rocm MAX_JOBS="${MAX_JOBS}" PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH}" \
    && export CMAKE_ARGS="-DGPU_TARGETS=${PYTORCH_ROCM_ARCH} -DAMDGPU_TARGETS=${PYTORCH_ROCM_ARCH}" \
    && python -m pip install --break-system-packages --no-build-isolation --no-deps -v -e . 2>&1 | tee /opt/r9700-vllm/build-info/vllm-build.log \
    && check-rocm-torch

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
    'torch', 'torchvision', 'torchaudio', 'triton', 'triton-kernels',
    'pytorch-triton-rocm', 'cuda-toolkit', 'cuda-bindings', 'cuda-pathfinder',
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

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/r9700-entrypoint"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]