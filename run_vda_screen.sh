#!/bin/bash
# Run VDA processing in a screen session - completely independent of VSCode
# Usage: bash run_vda_screen.sh
# Attach: screen -r vda_processing
# Detach: Ctrl+A then D

cd /home/akash_gemperts/gaussian-splatting

# Check if screen session already exists
if screen -list | grep -q "vda_processing"; then
    echo "Screen session 'vda_processing' already exists!"
    echo "To attach: screen -r vda_processing"
    echo "To kill existing: screen -X -S vda_processing quit"
    exit 1
fi

# Create screen session and run VDA processing
screen -dmS vda_processing bash -c '
    cd /home/akash_gemperts/gaussian-splatting
    source .venv/bin/activate
    
    echo "=========================================="
    echo "Starting VDA Processing in Screen Session"
    echo "=========================================="
    echo "Time: $(date)"
    echo ""
    
    cd Video-Depth-Anything
    
    # Start all 5 chunks
    echo "Launching GPU 0 - Chunk 00..."
    CUDA_VISIBLE_DEVICES=0 python run.py \
        --encoder vits \
        --input_video ../data/pantograph_scene/vda_video_chunks/chunk_00.mp4 \
        --output_dir ../data/pantograph_scene/vda_depth_final/chunk_00 \
        --save_npz \
        --grayscale \
        --fp32 \
        > ../data/pantograph_scene/vda_logs/chunk_00.log 2>&1 &
    
    echo "Launching GPU 1 - Chunk 01..."
    CUDA_VISIBLE_DEVICES=1 python run.py \
        --encoder vits \
        --input_video ../data/pantograph_scene/vda_video_chunks/chunk_01.mp4 \
        --output_dir ../data/pantograph_scene/vda_depth_final/chunk_01 \
        --save_npz \
        --grayscale \
        --fp32 \
        > ../data/pantograph_scene/vda_logs/chunk_01.log 2>&1 &
    
    echo "Launching GPU 2 - Chunk 02..."
    CUDA_VISIBLE_DEVICES=2 python run.py \
        --encoder vits \
        --input_video ../data/pantograph_scene/vda_video_chunks/chunk_02.mp4 \
        --output_dir ../data/pantograph_scene/vda_depth_final/chunk_02 \
        --save_npz \
        --grayscale \
        --fp32 \
        > ../data/pantograph_scene/vda_logs/chunk_02.log 2>&1 &
    
    echo "Launching GPU 3 - Chunk 03..."
    CUDA_VISIBLE_DEVICES=3 python run.py \
        --encoder vits \
        --input_video ../data/pantograph_scene/vda_video_chunks/chunk_03.mp4 \
        --output_dir ../data/pantograph_scene/vda_depth_final/chunk_03 \
        --save_npz \
        --grayscale \
        --fp32 \
        > ../data/pantograph_scene/vda_logs/chunk_03.log 2>&1 &
    
    echo "Launching GPU 4 - Chunk 04..."
    CUDA_VISIBLE_DEVICES=4 python run.py \
        --encoder vits \
        --input_video ../data/pantograph_scene/vda_video_chunks/chunk_04.mp4 \
        --output_dir ../data/pantograph_scene/vda_depth_final/chunk_04 \
        --save_npz \
        --grayscale \
        --fp32 \
        > ../data/pantograph_scene/vda_logs/chunk_04.log 2>&1 &
    
    echo ""
    echo "All 5 chunks started!"
    echo "Waiting for completion..."
    echo ""
    
    # Wait for all background jobs
    wait
    
    echo ""
    echo "=========================================="
    echo "VDA Processing Complete!"
    echo "=========================================="
    echo "Time: $(date)"
    echo ""
    echo "Output files:"
    ls -lh ../data/pantograph_scene/vda_depth_final/chunk_*/depth.npz 2>/dev/null || echo "ERROR: No NPZ files found!"
    echo ""
    echo "Press Enter to close this screen session..."
    read
'

echo "Screen session 'vda_processing' created and started!"
echo ""
echo "To monitor progress:"
echo "  screen -r vda_processing    # Attach to see live output"
echo "  Ctrl+A then D               # Detach (keeps running)"
echo ""
echo "Or check logs directly:"
echo "  tail -f data/pantograph_scene/vda_logs/chunk_*.log"
echo "  ./monitor_vda.sh"
