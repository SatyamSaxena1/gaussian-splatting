#!/bin/bash
# Run VDA one chunk at a time with immediate verification
# This ensures we get output even if processes are killed

cd /home/akash_gemperts/gaussian-splatting
source .venv/bin/activate
cd Video-Depth-Anything

for chunk in 00 01 02 03 04; do
    gpu=$((10#$chunk))  # Convert to decimal for GPU index
    
    echo "=========================================="
    echo "Processing Chunk $chunk on GPU $gpu"
    echo "Started: $(date)"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=$gpu python run.py \
        --encoder vits \
        --input_video ../data/pantograph_scene/vda_video_chunks/chunk_$chunk.mp4 \
        --output_dir ../data/pantograph_scene/vda_depth_final/chunk_$chunk \
        --save_npz \
        --grayscale \
        --fp32 \
        2>&1 | tee ../data/pantograph_scene/vda_logs/chunk_$chunk.log
    
    echo ""
    echo "Chunk $chunk completed at: $(date)"
    echo "Checking output..."
    ls -lh ../data/pantograph_scene/vda_depth_final/chunk_$chunk/*.npz 2>/dev/null || echo "WARNING: No NPZ file found!"
    echo ""
    
    # Small delay between chunks
    sleep 2
done

echo "=========================================="
echo "All chunks processed!"
echo "=========================================="
ls -lh ../data/pantograph_scene/vda_depth_final/chunk_*/chunk_*.npz
