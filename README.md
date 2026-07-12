# R9700 vLLM Docker AITER

Opinionated Docker build for running a custom ROCm/vLLM/AITER stack on an AMD Radeon AI PRO R9700 RDNA4 Linux host.

This repo exists because the current RDNA4 software path is still spicy. The goal is to turn the manual container setup into something repeatable and much easier to iterate on.

## What this builds

The image:

- uses the **ROCm 7.13.0 Technology Preview** userspace and development stack;
- installs AMD's official `gfx120X-all` TheRock tarball into `/opt/rocm`;
- installs the exact ROCm 7.13 PyTorch stack: `torch 2.11.0`, `torchvision 0.26.0`, and `torchaudio 2.11.0`;
- lets that PyTorch installation pull its matching ROCm 7.13 Triton dependency instead of replacing it from an older ROCm repository;
- uses system Python inside the image, with no venv;
- keeps Ubuntu's apt-installed Python build tooling in place to avoid `RECORD file not found` uninstall failures from `pip`, `wheel`, `setuptools`, and friends;
- builds AITER from a pinned commit;
- clones and builds vLLM directly from `magiccodingman/vllm-rdna4`;
- uses the `rdna4-dev` branch by default, with an optional `VLLM_REF` override;
- launches the installed `vllm serve` command directly;
- keeps Hugging Face, Torch, Triton, vLLM, and general pip caches under `/cache`.

## ROCm 7.13.0 Technology Preview

The default ROCm userspace comes from AMD's official RDNA4 tarball:

```dotenv
ROCM_BASE_IMAGE=ubuntu:24.04
ROCM_VERSION=7.13.0
ROCM_TARBALL_URL=https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx120X-all-7.13.0.tar.gz
```

The PyTorch installation uses AMD's `gfx120X-all` wheel repository and exact ROCm 7.13 packages:

```bash
python -m pip install \
  --break-system-packages \
  --force-reinstall \
  --ignore-installed \
  --no-cache-dir \
  --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ \
  "torch==2.11.0+rocm7.13.0" \
  "torchvision==0.26.0+rocm7.13.0" \
  "torchaudio==2.11.0+rocm7.13.0"
```

The Docker build verifies all of the following before continuing:

```text
torch == 2.11.0+rocm7.13.0
rocm package == 7.13.0
torch.version.hip is in the 7.13.x release family
```

AMD's PyTorch wheel currently reports the HIP build as a more specific internal value such as `7.13.99004`, so the validator checks the exact ROCm package version and the HIP `7.13` release family instead of requiring the runtime build string to equal `7.13.0` literally.

The resolved ROCm version is also recorded inside the image at:

```text
/opt/r9700-vllm/build-info/rocm.version.txt
```

## Host ROCm driver reality check

The Docker image includes ROCm userspace. It does **not** and cannot fully install the host AMDGPU/KFD kernel driver from inside the container.

The host Linux machine still needs working GPU device access:

```bash
/dev/kfd
/dev/dri
```

Run this first on the R9700 host:

```bash
python3 scripts/host-rocm-check.py
```

or:

```bash
./scripts/host-rocm-check.sh
```

If that fails, fix the host ROCm/AMDGPU driver first. Once `/dev/kfd` and `/dev/dri` are present, the Docker image owns the ROCm 7.13.0 Technology Preview userspace side.

## First-time setup

```bash
cp .env.example .env

sed -i "s/^VIDEO_GID=.*/VIDEO_GID=$(getent group video | cut -d: -f3)/" .env
sed -i "s/^RENDER_GID=.*/RENDER_GID=$(getent group render | cut -d: -f3)/" .env

nano .env
```

At minimum, set the host model directory:

```bash
MODEL_DIR=/mnt/fastdisk/ai-models
```

`MODEL_DIR` is the host path. It mounts into the container as `/models`.

Do **not** put the model path in `.env`. The actual model is a vLLM runtime argument.

## Select the vLLM source ref

The default source is:

```dotenv
VLLM_REPO=https://github.com/magiccodingman/vllm-rdna4.git
VLLM_REF=rdna4-dev
```

`VLLM_REF` is passed to `git checkout`, so it may be changed to `main`, another branch, a tag, or a commit before building.

For example:

```bash
VLLM_REF=main python3 scripts/r9700-vllm.py build
```

or set it in `.env`:

```dotenv
VLLM_REF=main
```

No external vLLM PR branches are fetched, merged, or cherry-picked by this repository. All required RDNA4 code changes are expected to already exist in the selected ref of `vllm-rdna4`.

## Build

```bash
python3 scripts/r9700-vllm.py build
```

or directly:

```bash
docker compose build
```

The selected vLLM ref and resolved commit are recorded inside the image under:

```text
/opt/r9700-vllm/build-info/vllm.ref.txt
/opt/r9700-vllm/build-info/vllm.commit.txt
```

## Launch a model

Use the helper and pass vLLM arguments normally:

```bash
python3 scripts/r9700-vllm.py serve --model /models/YourModel --tensor-parallel-size 2
```

There is also a convenience positional form:

```bash
python3 scripts/r9700-vllm.py serve /models/YourModel --tensor-parallel-size 2
```

Both forms launch the installed `vllm serve` command directly, start the server detached, remove any old `r9700-vllm` container first, and keep the service ports enabled.

Foreground one-shot launch:

```bash
python3 scripts/r9700-vllm.py serve-fg --model /models/YourModel --tensor-parallel-size 2
```

Follow logs:

```bash
python3 scripts/r9700-vllm.py logs
```

Verify ROCm/PyTorch visibility inside the running container:

```bash
python3 scripts/r9700-vllm.py verify
```

Open a shell:

```bash
python3 scripts/r9700-vllm.py shell
```

Stop/restart/clean:

```bash
python3 scripts/r9700-vllm.py stop
python3 scripts/r9700-vllm.py restart
python3 scripts/r9700-vllm.py clean
```

The tiny shell/cmd wrappers exist only for convenience:

```bash
./scripts/r9700-vllm serve --model /models/YourModel --tensor-parallel-size 2
```

The source of truth is `scripts/r9700-vllm.py`.

## Swap models

Stop and replace the current model by launching again:

```bash
python3 scripts/r9700-vllm.py serve --model /models/NewModel --tensor-parallel-size 2 --max-model-len 8192
```

The helper removes the old container before starting the new one.

## Runtime flags

Runtime behavior comes from the environment and the arguments passed to `vllm serve`.

For example:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
python3 scripts/r9700-vllm.py serve \
  --model /models/YourModel \
  --tensor-parallel-size 2
```

Stable path/cache/compiler basics remain in the Docker image. Model-specific flags should be supplied at runtime rather than baked into the image.

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

## Clean up an old failed run/build

Safe runtime cleanup:

```bash
python3 scripts/r9700-vllm.py clean
```

That removes the named container and Compose orphans. It does not delete the image or caches.

To also remove the image tag:

```bash
python3 scripts/r9700-vllm.py nuke-image
```

If Docker build cache got especially stale, use Docker's builder prune manually:

```bash
docker builder prune
```

Do not prune caches unless you actually want to redownload/rebuild everything.

## File layout

```text
Dockerfile
.dockerignore
docker-compose.yml
.env.example
docker/
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