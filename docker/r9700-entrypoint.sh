#!/usr/bin/env bash
set -euo pipefail

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"

# This is a ROCm-only image. Ensure host or Docker environment inheritance does
# not make vLLM detect a CUDA visibility configuration and emit warnings.
unset CUDA_VISIBLE_DEVICES

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache}"
export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/pip}"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" /logs

has_model_arg() {
  local arg
  for arg in "$@"; do
    if [ "$arg" = "--model" ] || [[ "$arg" == --model=* ]]; then
      return 0
    fi
  done
  return 1
}

mode="${1:-serve}"
case "$mode" in
  serve)
    shift || true
    if ! has_model_arg "$@"; then
      cat >&2 <<'EOF'
[entrypoint] No vLLM model was provided.

Pass the model at launch time instead of baking it into .env, for example:

  python scripts/r9700-vllm.py serve --model /models/YourModel --tensor-parallel-size 2

or:

  python scripts/r9700-vllm.py serve /models/YourModel --tensor-parallel-size 2
EOF
      exit 2
    fi

    /usr/local/bin/verify-rocm.py --warn-only
    exec vllm serve "$@"
    ;;
  verify)
    shift || true
    exec /usr/local/bin/verify-rocm.py "$@"
    ;;
  shell|bash)
    shift || true
    exec /bin/bash "$@"
    ;;
  python)
    shift || true
    exec python "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
