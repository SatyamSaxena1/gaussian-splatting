#!/bin/bash
set -e

# Configuration
PROJECT_DIR="/home/akash_gemperts/gaussian-splatting"
VENV_PATH="$PROJECT_DIR/.venv"
VDA_DIR="$PROJECT_DIR/Video-Depth-Anything"
INPUT_VIDEO="$PROJECT_DIR/data/whatsapp_test/input.mp4"
OUTPUT_DIR="$PROJECT_DIR/data/whatsapp_test/vda_depth_raw"

echo "Running VDA on WhatsApp clip..."
source "$VENV_PATH/bin/activate"
cd "$VDA_DIR"

mkdir -p "$OUTPUT_DIR"

# Run on GPU 0
CUDA_VISIBLE_DEVICES=0 python run.py \
    --encoder vits \
    --input_video "$INPUT_VIDEO" \
    --output_dir "$OUTPUT_DIR" \
    --save_npz \
    --grayscale \
    --fp32

echo "Done."
