#!/bin/bash
# Wrapper script to run COLMAP inside micromamba environment
# while respecting CUDA_VISIBLE_DEVICES and COLMAP_GPU environment variables

# Activate micromamba environment if not already activated
if [ -z "$MAMBA_EXE" ]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate gaussian_splatting
fi

# Pass through CUDA_VISIBLE_DEVICES to respect GPU selection
# COLMAP_GPU can be set to specify which GPU index to use
exec colmap "$@"
