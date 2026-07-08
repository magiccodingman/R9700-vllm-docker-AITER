#!/usr/bin/env python3
from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    system = platform.system().lower()
    if system != "linux":
        print(f"Host ROCm GPU passthrough check: {platform.system()} detected.")
        print("This project is intentionally for an R9700 Linux ROCm Docker host.")
        print("Non-Linux machines can control Docker only if their Docker context points at that Linux host.")
        return 1

    ok = True
    for p in [Path("/dev/kfd"), Path("/dev/dri")]:
        if p.exists():
            print(f"OK: {p} exists")
        else:
            print(f"MISSING: {p}")
            ok = False

    if shutil.which("rocminfo"):
        rc = run(["rocminfo"])
        ok = ok and (rc == 0)
    else:
        print("WARN: rocminfo not found on host PATH. Device nodes may still be usable by the container.")

    if not ok:
        print("\nHost ROCm access is not ready. Install/fix the host AMDGPU/KFD driver first, then rerun this check.")
        return 1

    print("\nHost ROCm device access looks ready for Docker passthrough.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
