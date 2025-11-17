#!/bin/bash
# Run VDA processing in tmux - more robust than screen
# Usage: bash run_vda_tmux.sh
# Attach: tmux attach -t vda
# Detach: Ctrl+B then D

cd /home/akash_gemperts/gaussian-splatting

# Check if tmux session exists
if tmux has-session -t vda 2>/dev/null; then
    echo "Tmux session 'vda' already exists!"
    echo "To attach: tmux attach -t vda"
    echo "To kill: tmux kill-session -t vda"
    exit 1
fi

# Create tmux session
tmux new-session -d -s vda "bash -c '
    cd /home/akash_gemperts/gaussian-splatting
    source .venv/bin/activate
    cd Video-Depth-Anything
    
    echo \"===========================================\"
    echo \"VDA Processing Started: \$(date)\"
    echo \"===========================================\"
    echo \"\"
    
    # Launch all 5 GPUs
    CUDA_VISIBLE_DEVICES=0 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_00.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_00 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_00.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_01.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_01 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_01.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_02.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_02 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_02.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_03.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_03 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_03.log 2>&1 &
    CUDA_VISIBLE_DEVICES=4 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_04.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_04 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_04.log 2>&1 &
    
    echo \"All 5 GPU processes launched\"
    echo \"Waiting for completion...\"
    echo \"\"
    
    wait
    
    echo \"\"
    echo \"===========================================\"
    echo \"VDA Processing Complete: \$(date)\"
    echo \"===========================================\"
    ls -lh ../data/pantograph_scene/vda_depth_final/chunk_*/depth.npz 2>/dev/null || echo \"ERROR: No NPZ files\"
    echo \"\"
    echo \"Session will remain open. Press Ctrl+B then D to detach.\"
    bash
'"

echo "Tmux session 'vda' created!"
echo ""
echo "Commands:"
echo "  tmux attach -t vda      # View live output"
echo "  Ctrl+B then D           # Detach (keeps running)"
echo "  tail -f data/pantograph_scene/vda_logs/chunk_00.log"
