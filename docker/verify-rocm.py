#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description="Verify ROCm device passthrough and PyTorch visibility inside the container.")
parser.add_argument("--warn-only", action="store_true", help="Print warnings instead of exiting non-zero.")
args = parser.parse_args()

problems = []


def note(msg: str) -> None:
    print(f"[rocm-check] {msg}", flush=True)


def require(path: str, what: str) -> None:
    if not os.path.exists(path):
        problems.append(f"missing {what}: {path}")
    else:
        note(f"found {what}: {path}")


require("/dev/kfd", "KFD compute device")
require("/dev/dri", "DRI device directory")

if shutil.which("rocminfo"):
    try:
        out = subprocess.run(["rocminfo"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        lines = [ln for ln in out.stdout.splitlines() if "Name:" in ln or "gfx" in ln]
        note("rocminfo sample:")
        for ln in lines[:25]:
            print(f"  {ln}")
        if out.returncode != 0:
            problems.append("rocminfo returned non-zero")
    except Exception as exc:
        problems.append(f"rocminfo failed: {exc}")
else:
    problems.append("rocminfo not found in image")

try:
    import torch
    note(f"torch: {torch.__version__}")
    note(f"torch file: {torch.__file__}")
    note(f"HIP: {torch.version.hip}")
    count = torch.cuda.device_count()
    note(f"torch.cuda.device_count(): {count}")
    for i in range(count):
        note(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    if count == 0:
        problems.append("PyTorch sees zero ROCm/CUDA devices")
except Exception as exc:
    problems.append(f"PyTorch ROCm verification failed: {exc}")

if problems:
    note("problems detected:")
    for p in problems:
        print(f"  - {p}")
    if not args.warn_only:
        sys.exit(1)
else:
    note("ROCm container visibility looks good.")
