#!/bin/bash
# Example usage of Video Depth Anything multi-GPU pipeline

# Configuration
VIDEO_PATH="data/pantograph_scene/input.mp4"
CHUNKS_DIR="data/pantograph_scene/vda_chunks"
DEPTH_DIR="data/pantograph_scene/vda_depth_chunks"
MERGED_DIR="data/pantograph_scene/vda_merged_depth"
VDA_PATH="./Video-Depth-Anything"
NUM_GPUS=5
MODEL="vits"

# Activate environment
source .venv/bin/activate

echo "============================================================"
echo "Video Depth Anything Multi-GPU Pipeline"
echo "============================================================"
echo "Video: $VIDEO_PATH"
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL"
echo "============================================================"

# Step 1: Split video into chunks
echo -e "\nStep 1/3: Splitting video into chunks..."
python tools/split_video_chunks.py \
    --input "$VIDEO_PATH" \
    --output_dir "$CHUNKS_DIR" \
    --num_gpus $NUM_GPUS \
    --overlap 32

# Step 2: Process chunks in parallel
echo -e "\nStep 2/3: Processing chunks on multiple GPUs..."
python tools/process_chunks_parallel.py \
    --metadata "$CHUNKS_DIR/chunks_metadata.json" \
    --vda_path "$VDA_PATH" \
    --output_dir "$DEPTH_DIR" \
    --gpu_ids 0 1 2 3 4 \
    --model_size "$MODEL"

# Step 3: Merge depth chunks
echo -e "\nStep 3/3: Merging depth chunks..."
python tools/merge_depth_chunks.py \
    --metadata "$CHUNKS_DIR/chunks_metadata.json" \
    --depth_dir "$DEPTH_DIR" \
    --output_dir "$MERGED_DIR" \
    --format png16 \
    --create_video

echo -e "\n============================================================"
echo "Pipeline complete!"
echo "Merged depth maps: $MERGED_DIR"
echo "============================================================"
