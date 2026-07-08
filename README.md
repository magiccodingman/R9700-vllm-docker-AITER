# R9700 vLLM Docker AITER

Opinionated Docker build for running a custom ROCm/vLLM/AITER stack on an AMD Radeon RX 9700-class RDNA4 Linux host.

This repo exists because the current RDNA4 software path is still spicy. The goal is to turn the manual container setup into something repeatable, pinned, and much easier to iterate on.

## What this builds

The image:

- uses ROCm 7.2.4 userspace by default via `rocm/dev-ubuntu-24.04:7.2.4-complete`;
- installs PyTorch from the ROCm 7.2 wheel index instead of using `rocm/pytorch-nightly:nightly`;
- uses system Python inside the image, with no venv;
- builds AITER from a pinned commit;
- builds vLLM from a pinned base commit plus the RDNA4/R9700 patch stack;
- pulls `launch_vllm.py` from `magiccodingman/VllmLaunchScriptR9700`;
- leaves vLLM/AITER runtime flags to `launch_vllm.py` instead of baking them into Docker;
- keeps Hugging Face, Torch, Triton, vLLM, and pip caches under `/cache`.

## Host ROCm driver reality check

The Docker image includes ROCm userspace. It does **not** and cannot fully install the host AMDGPU/KFD kernel driver from inside the container.

The host Linux machine still needs working GPU device access:

```bash
/dev/kfd
/dev/dri
```

The container then receives those devices through Docker Compose.

Run this first on the R9700 host:

```bash
python scripts/host-rocm-check.py
```

or:

```bash
./scripts/host-rocm-check.sh
```

If that fails, fix the host ROCm/AMDGPU driver first. Once `/dev/kfd` and `/dev/dri` are present, the Docker image owns the ROCm 7.2.4 userspace side.

## First-time setup

```bash
cp .env.example .env

sed -i "s/^VIDEO_GID=.*/VIDEO_GID=$(getent group video | cut -d: -f3)/" .env
sed -i "s/^RENDER_GID=.*/RENDER_GID=$(getent group render | cut -d: -f3)/" .env

nano .env
```

At minimum, set:

```bash
MODEL_DIR=/mnt/fastdisk/ai-models
MODEL_PATH=/models/YourModelFolderOrFile
```

`MODEL_DIR` is the host path. `MODEL_PATH` is the in-container path after `MODEL_DIR` mounts as `/models`.

## Build

```bash
docker compose build
```

or:

```bash
docker build -t r9700-vllm:rocm724 .
```

The vLLM patch-stack script verifies fork branch tips before cherry-picking. That keeps the build pinned instead of silently following moving PR branches.

## Run

```bash
docker compose up -d
```

Follow logs:

```bash
python scripts/r9700-vllm.py logs
```

Open a shell:

```bash
python scripts/r9700-vllm.py shell
```

Verify ROCm/PyTorch visibility inside the running container:

```bash
python scripts/r9700-vllm.py verify
```

Stop/restart/recreate:

```bash
python scripts/r9700-vllm.py stop
python scripts/r9700-vllm.py restart
python scripts/r9700-vllm.py recreate
```

The tiny shell/cmd wrappers exist only for convenience:

```bash
./scripts/r9700-vllm logs
```

The source of truth is `scripts/r9700-vllm.py`.

## Swap models

Edit `.env`:

```bash
MODEL_PATH=/models/NewModel
```

Then recreate:

```bash
python scripts/r9700-vllm.py recreate
```

Or override for a single command:

```bash
MODEL_PATH=/models/NewModel docker compose up -d --force-recreate
```

## Runtime flags

Do not put the vLLM/AITER runtime tuning flags in the Dockerfile unless they are truly container plumbing.

Put model/runtime behavior in `launch_vllm.py`, for example:

```python
os.environ["VLLM_ROCM_USE_AITER"] = "1"
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "FALSE"
os.environ["VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION"] = "1"
```

The Dockerfile only sets stable path/cache/compiler basics.

To use a local launch script without rebuilding, uncomment this in `docker-compose.yml`:

```yaml
# - ./launch_vllm.py:/opt/r9700-vllm/launcher/launch_vllm.py:ro
```

## API access

The Compose file uses host networking and exposes vLLM on port `8000` by default.

Health-ish check:

```bash
curl http://127.0.0.1:8000/v1/models
```

Example OpenAI-compatible request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "whatever-vllm-reports",
    "messages": [{"role": "user", "content": "Say hello from the R9700."}]
  }'
```

## File layout

```text
Dockerfile
.dockerignore
docker-compose.yml
.env.example
docker/
  apply-vllm-stack.sh
  r9700-entrypoint.sh
  verify-rocm.py
scripts/
  host-rocm-check.py
  host-rocm-check.sh
  host-rocm-check.cmd
  r9700-vllm.py
  r9700-vllm
  r9700-vllm.cmd
```

## Notes

This is intentionally R9700/Linux-first. The Python helper can run on other operating systems if they are controlling a Docker context, but the actual ROCm GPU host is expected to be Linux with `/dev/kfd` and `/dev/dri`.
