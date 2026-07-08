#!/usr/bin/env bash
set -euo pipefail

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache}"
export HF_HOME="${HF_HOME:-/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/pip}"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" /logs

mode="${1:-serve}"
case "$mode" in
  serve)
    shift || true
    /usr/local/bin/verify-rocm.py --warn-only
    launch_script="${VLLM_LAUNCH_SCRIPT:-/opt/r9700-vllm/launcher/launch_vllm.py}"
    if [ -f "$launch_script" ]; then
      echo "[entrypoint] launching through: $launch_script"
      exec python "$launch_script" "$@"
    fi
    echo "[entrypoint] launch script not found; falling back to vLLM OpenAI API server module."
    exec python -m vllm.entrypoints.openai.api_server "$@"
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
