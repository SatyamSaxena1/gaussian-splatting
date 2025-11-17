#!/bin/bash
set -euo pipefail
NVIDIA_LIB_ROOT="$(pwd)/NVIDIA-Linux-x86_64-580.65.06"
export LD_LIBRARY_PATH="${NVIDIA_LIB_ROOT}:${LD_LIBRARY_PATH:-}"
export PATH="${NVIDIA_LIB_ROOT}:${PATH}"
exec "$@"
