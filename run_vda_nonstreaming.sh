#!/bin/bash
# Run all 5 video chunks in parallel on separate GPUs using run.py
# This saves depth as NPZ format (compressed numpy arrays)

cd /home/akash_gemperts/gaussian-splatting/Video-Depth-Anything
source ../.venv/bin/activate

echo "Starting VDA processing on 5 GPUs (run.py with --save_npz)..."

# Chunk 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python run.py \
    --input_video ../data/pantograph_scene/vda_video_chunks/chunk_00.mp4 \
    --output_dir ../data/pantograph_scene/vda_depth_final/chunk_00 \
    --encoder vits \
    --save_npz \
    --grayscale \
    --fp32 \
    > ../data/pantograph_scene/vda_logs/chunk_00.log 2>&1 &

# Chunk 1 on GPU 1  
CUDA_VISIBLE_DEVICES=1 python run.py \
    --input_video ../data/pantograph_scene/vda_video_chunks/chunk_01.mp4 \
    --output_dir ../data/pantograph_scene/vda_depth_final/chunk_01 \
    --encoder vits \
    --save_npz \
    --grayscale \
    --fp32 \
    > ../data/pantograph_scene/vda_logs/chunk_01.log 2>&1 &

# Chunk 2 on GPU 2
CUDA_VISIBLE_DEVICES=2 python run.py \
    --input_video ../data/pantograph_scene/vda_video_chunks/chunk_02.mp4 \
    --output_dir ../data/pantograph_scene/vda_depth_final/chunk_02 \
    --encoder vits \
    --save_npz \
    --grayscale \
    --fp32 \
    > ../data/pantograph_scene/vda_logs/chunk_02.log 2>&1 &

# Chunk 3 on GPU 3
CUDA_VISIBLE_DEVICES=3 python run.py \
    --input_video ../data/pantograph_scene/vda_video_chunks/chunk_03.mp4 \
    --output_dir ../data/pantograph_scene/vda_depth_final/chunk_03 \
    --encoder vits \
    --save_npz \
    --grayscale \
    --fp32 \
    > ../data/pantograph_scene/vda_logs/chunk_03.log 2>&1 &

# Chunk 4 on GPU 4
CUDA_VISIBLE_DEVICES=4 python run.py \
    --input_video ../data/pantograph_scene/vda_video_chunks/chunk_04.mp4 \
    --output_dir ../data/pantograph_scene/vda_depth_final/chunk_04 \
    --encoder vits \
    --save_npz \
    --grayscale \
    --fp32 \
    > ../data/pantograph_scene/vda_logs/chunk_04.log 2>&1 &

echo "All 5 chunks started in background"
echo "Monitor logs: tail -f ../data/pantograph_scene/vda_logs/chunk_*.log"
echo "GPU usage: watch -n 2 nvidia-smi"
echo ""
echo "Note: run.py loads entire video into memory. With ~1,300 frames per chunk"
echo "and 6GB VRAM per GPU, this should work for the vits model."
