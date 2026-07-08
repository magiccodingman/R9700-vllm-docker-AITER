#!/usr/bin/env python3
"""
Single-source control helper for the R9700 vLLM Docker service.

This shells out to Docker/Docker Compose. The actual ROCm GPU runtime still
requires a Linux Docker host exposing /dev/kfd and /dev/dri.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

DEFAULT_CONTAINER = "r9700-vllm"
DEFAULT_COMPOSE_FILE = "docker-compose.yml"


def env(name: str, default: str) -> str:
    return os.environ.get(name, os.environ.get(name.replace("R9700_", ""), default))


def die(message: str, code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def find_compose_cmd() -> list[str]:
    explicit = os.environ.get("R9700_COMPOSE_CMD")
    if explicit:
        return explicit.split()

    docker = shutil.which("docker")
    if not docker:
        die("Docker CLI was not found on PATH.")

    rc = subprocess.run([docker, "compose", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc == 0:
        return [docker, "compose"]

    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]

    die("Neither 'docker compose' nor 'docker-compose' is available.")


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def usage() -> None:
    print(
        """Usage:
  python scripts/r9700-vllm.py <command> [args...]

Commands:
  up                   Start/recreate service with docker compose up -d
  build                Build image with docker compose build
  logs                 Follow container logs
  shell                Open bash inside the container
  verify               Run ROCm/PyTorch visibility checks inside the container
  exec <cmd...>        Run any command inside the container
  stop                 Stop the running container
  down                 docker compose down
  restart              Restart the running container
  recreate             docker compose up -d --force-recreate
  ps                   Show container status
  status               Show docker compose service status

Environment:
  R9700_CONTAINER      Container name override, default r9700-vllm
  CONTAINER_NAME       Also accepted for compose/.env compatibility
  R9700_COMPOSE_FILE   Compose file override, default docker-compose.yml
  COMPOSE_FILE         Also accepted
  R9700_COMPOSE_CMD    Compose command override, e.g. "docker compose"

Examples:
  python scripts/r9700-vllm.py build
  python scripts/r9700-vllm.py up
  python scripts/r9700-vllm.py logs
  python scripts/r9700-vllm.py exec python -c "import torch; print(torch.cuda.device_count())"
""".rstrip()
    )


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"
    rest = argv[2:]

    container = env("R9700_CONTAINER", env("CONTAINER_NAME", DEFAULT_CONTAINER))
    compose_file = env("R9700_COMPOSE_FILE", env("COMPOSE_FILE", DEFAULT_COMPOSE_FILE))
    compose = find_compose_cmd() if cmd not in {"help", "-h", "--help"} else []

    if cmd in {"help", "-h", "--help"}:
        usage()
        return 0
    if cmd == "build":
        return run([*compose, "-f", compose_file, "build", *rest])
    if cmd == "up":
        return run([*compose, "-f", compose_file, "up", "-d", *rest])
    if cmd == "recreate":
        return run([*compose, "-f", compose_file, "up", "-d", "--force-recreate", *rest])
    if cmd == "down":
        return run([*compose, "-f", compose_file, "down", *rest])
    if cmd == "status":
        return run([*compose, "-f", compose_file, "ps", *rest])
    if cmd == "logs":
        return run(["docker", "logs", "-f", container, *rest])
    if cmd == "shell":
        return run(["docker", "exec", "-it", container, "bash", *rest])
    if cmd == "verify":
        return run(["docker", "exec", "-it", container, "r9700-entrypoint", "verify", *rest])
    if cmd == "exec":
        if not rest:
            die("exec requires a command, e.g. python scripts/r9700-vllm.py exec bash")
        return run(["docker", "exec", "-it", container, *rest])
    if cmd == "stop":
        return run(["docker", "stop", container, *rest])
    if cmd == "restart":
        return run(["docker", "restart", container, *rest])
    if cmd == "ps":
        return run(["docker", "ps", "--filter", f"name={container}", *rest])

    die(f"Unknown command: {cmd}. Run: python scripts/r9700-vllm.py help")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
