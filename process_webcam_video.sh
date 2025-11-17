#!/bin/bash

# Quick script to process the webcam video for Gaussian Splatting

set -euo pipefail

VIDEO=${VIDEO:-""}
OUTPUT_NAME=${OUTPUT_NAME:-"webcam_scene"}
FPS=${FPS:-2}  # frames per second
RUN_COLMAP=${RUN_COLMAP:-0}
RUN_TRAIN=${RUN_TRAIN:-0}
BAR_WIDTH=40

usage() {
	cat <<'EOF'
Usage: process_webcam_video.sh [--video path.mp4] [--output-name scene_name] [--fps N] [--run-colmap] [--run-train]

Environment variables (override defaults):
  VIDEO=...          Absolute path to the source video (mp4/mov/etc).
  OUTPUT_NAME=...    Short name for the scene folder inside data/ (default: webcam_scene).
  FPS=...            Extraction rate for ffmpeg (default: 2).
  RUN_COLMAP=1       Actually launch convert.py (requires GPU + escalated shell).
  RUN_TRAIN=1        Launch train.py using staged CUDA libs (requires GPU + escalated shell).

Examples:
  VIDEO=/path/movie.mp4 OUTPUT_NAME=living_room ./process_webcam_video.sh
  ./process_webcam_video.sh --video /data/clip.mp4 --output-name couch --fps 3
EOF
}

while (( "$#" )); do
	case "$1" in
		--video)
			VIDEO=$2; shift 2;;
		--output-name)
			OUTPUT_NAME=$2; shift 2;;
		--fps)
			FPS=$2; shift 2;;
		--run-colmap)
			RUN_COLMAP=1; shift;;
		--run-train)
			RUN_TRAIN=1; shift;;
		-h|--help)
			usage; exit 0;;
		*)
			echo "Unknown argument: $1"; usage; exit 1;;
	esac
done

if [ -z "$VIDEO" ]; then
	echo "ERROR: VIDEO path not provided. Set VIDEO env or pass --video."
	usage
	exit 1
fi

if [ ! -f "$VIDEO" ]; then
	echo "ERROR: VIDEO file not found: $VIDEO"
	exit 1
fi

OUTPUT_ROOT="/home/akash_gemperts/gaussian-splatting/data"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}"
BAR_WIDTH=40

echo "=========================================="
echo "Processing Webcam Video for Gaussian Splatting"
echo "=========================================="
echo ""
echo "Video: $VIDEO"
echo "Output folder: $OUTPUT_DIR"
echo "Extraction rate: $FPS fps"
echo "Run COLMAP automatically: ${RUN_COLMAP}"
echo "Run training automatically: ${RUN_TRAIN}"
echo ""

draw_progress_bar() {
	local current=$1
	local total=$2
	local width=${3:-40}

	if [ "$total" -le 0 ]; then
		total=1
	fi

	if [ "$current" -gt "$total" ]; then
		current=$total
	fi

	local percent=$((current * 100 / total))
	local filled=$((current * width / total))
	local empty=$((width - filled))

	if [ "$filled" -gt 0 ]; then
		printf -v bar '%*s' "$filled" ''
		bar=${bar// /#}
	else
		bar=""
	fi

	if [ "$empty" -gt 0 ]; then
		printf -v spaces '%*s' "$empty" ''
		spaces=${spaces// /-}
	else
		spaces=""
	fi

	printf "\r[%s%s] %3d%% (%d/%d)" "$bar" "$spaces" "$percent" "$current" "$total"
}

# Create directory
mkdir -p "$OUTPUT_DIR/input"

echo "Step 1: Extracting frames (this will take a few minutes)..."
echo "Video is ~20 minutes, extracting at $FPS fps = ~2400 frames"
echo ""

# Determine total frames for progress tracking
TOTAL_FRAMES=0
if command -v ffprobe >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
	VIDEO_DURATION=$(ffprobe -i "$VIDEO" -show_entries format=duration -v quiet -of csv="p=0")
	if [ -n "$VIDEO_DURATION" ] && [ "$VIDEO_DURATION" != "N/A" ]; then
	TOTAL_FRAMES=$(VIDEO_DURATION="$VIDEO_DURATION" FPS="$FPS" python3 - <<'PY'
import math
import os

duration = os.environ.get('VIDEO_DURATION', '0')
fps = os.environ.get('FPS', '0')

try:
	duration_value = float(duration)
	fps_value = float(fps)
except ValueError:
	print(0)
else:
	total = math.ceil(duration_value * fps_value)
	print(int(total))
PY
)
	fi
fi

if [ "$TOTAL_FRAMES" -gt 0 ]; then
	echo "Progress:"
	PROGRESS_PIPE=$(mktemp -u)
	mkfifo "$PROGRESS_PIPE"

	cleanup_progress() {
		rm -f "$PROGRESS_PIPE"
	}

	trap cleanup_progress EXIT

	ffmpeg -i "$VIDEO" -vf "fps=$FPS,scale=960:540" -qscale:v 2 "$OUTPUT_DIR/input/frame_%04d.jpg" -progress "$PROGRESS_PIPE" -nostats -loglevel error &
	FFMPEG_PID=$!

	CURRENT_FRAME=0
	while IFS='=' read -r key value; do
		case "$key" in
			frame)
				CURRENT_FRAME=${value:-0}
				draw_progress_bar "$CURRENT_FRAME" "$TOTAL_FRAMES" "$BAR_WIDTH"
				;;
			progress)
				if [ "$value" = "end" ]; then
					break
				fi
				;;
		esac
	done < "$PROGRESS_PIPE"

	wait "$FFMPEG_PID"
	printf "\n"
	cleanup_progress
	trap - EXIT
else
	echo "ffprobe/python3 not available or video duration unknown; running without progress bar."
	# Extract frames at 1/2 resolution for faster processing
	ffmpeg -i "$VIDEO" -vf "fps=$FPS,scale=960:540" -qscale:v 2 "$OUTPUT_DIR/input/frame_%04d.jpg" -hide_banner
fi

FRAME_COUNT=$(ls -1 "$OUTPUT_DIR/input"/*.jpg 2>/dev/null | wc -l)
echo ""
echo "✓ Extracted $FRAME_COUNT frames to $OUTPUT_DIR/input/"
echo ""
echo "=========================================="
echo "Next steps"
echo "=========================================="
echo ""

COLMAP_CMD="COLMAP_GPU=0 XDG_CACHE_HOME=$PWD/.mamba-cache python3 convert.py -s \"$OUTPUT_DIR\" \
  --colmap_executable \"$PWD/run_colmap_local.sh\" \
  --matching sequential \
  --sequential_overlap 3 \
  --sift_max_image_size 1600 \
  --sift_gpu_index 0 \
  --sift_num_threads 8"

TRAIN_CMD="./gpu_env.sh python3 train.py -s \"$OUTPUT_DIR\" --data_device cuda"

if [ "$RUN_COLMAP" -eq 1 ]; then
	echo "Launching COLMAP end-to-end..."
	eval "$COLMAP_CMD"
else
	echo "To run COLMAP (requires CUDA + escalated shell):"
	echo "  $COLMAP_CMD"
fi

COLMAP_OUT="$OUTPUT_DIR/distorted/sparse/0"
if [ -d "$COLMAP_OUT" ] && [ -f "$COLMAP_OUT/cameras.bin" ] && [ -f "$COLMAP_OUT/images.bin" ] && [ -f "$COLMAP_OUT/points3D.bin" ]; then
	echo "✓ Existing COLMAP output found: $COLMAP_OUT"
else
	echo "ℹ️  COLMAP output not detected yet; run the command above once GPU access is available."
fi

if [ "$RUN_TRAIN" -eq 1 ]; then
	echo "Starting training..."
	eval "$TRAIN_CMD"
else
	echo "To start training (after COLMAP succeeds):"
	echo "  export LD_LIBRARY_PATH=/home/akash_gemperts/.local/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
	echo "  $TRAIN_CMD"
fi

echo ""
echo "When training completes, render/inspect with:"
echo "  ./gpu_env.sh python3 render.py --model_path \"$OUTPUT_DIR\" --iteration 30000"
