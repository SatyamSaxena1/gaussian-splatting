#!/bin/bash
# Fully detached VDA processing that survives terminal/VSCode crashes

cd /home/akash_gemperts/gaussian-splatting

# Activate virtual environment
source .venv/bin/activate

# Start all 5 chunks in fully detached mode
setsid bash -c "cd /home/akash_gemperts/gaussian-splatting/Video-Depth-Anything && CUDA_VISIBLE_DEVICES=0 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_00.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_00 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_00.log 2>&1" &

setsid bash -c "cd /home/akash_gemperts/gaussian-splatting/Video-Depth-Anything && CUDA_VISIBLE_DEVICES=1 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_01.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_01 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_01.log 2>&1" &

setsid bash -c "cd /home/akash_gemperts/gaussian-splatting/Video-Depth-Anything && CUDA_VISIBLE_DEVICES=2 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_02.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_02 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_02.log 2>&1" &

setsid bash -c "cd /home/akash_gemperts/gaussian-splatting/Video-Depth-Anything && CUDA_VISIBLE_DEVICES=3 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_03.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_03 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_03.log 2>&1" &

setsid bash -c "cd /home/akash_gemperts/gaussian-splatting/Video-Depth-Anything && CUDA_VISIBLE_DEVICES=4 python run.py --encoder vits --input_video ../data/pantograph_scene/vda_video_chunks/chunk_04.mp4 --output_dir ../data/pantograph_scene/vda_depth_final/chunk_04 --save_npz --grayscale --fp32 > ../data/pantograph_scene/vda_logs/chunk_04.log 2>&1" &

echo "All 5 chunks started in detached sessions with setsid"
echo "These will survive terminal/VSCode crashes"
echo "Monitor: tail -f data/pantograph_scene/vda_logs/chunk_*.log"
