#!/bin/bash
set -euo pipefail
export MAMBA_ROOT_PREFIX="/home/akash_gemperts/micromamba"
export XDG_CACHE_HOME="$(pwd)/.mamba-cache"
mkdir -p "$XDG_CACHE_HOME"
exec /home/akash_gemperts/bin/micromamba run -n colmap colmap "$@"
