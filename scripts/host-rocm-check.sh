#!/usr/bin/env sh
# Optional POSIX shim. The real helper is host-rocm-check.py.
exec python3 "$(dirname "$0")/host-rocm-check.py" "$@"
