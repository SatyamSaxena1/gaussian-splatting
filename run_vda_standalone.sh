#!/bin/bash
################################################################################
# VDA Processing Script - Run Standalone
# This script runs VDA depth estimation completely independently
# No VSCode, no background processes that can be killed
################################################################################

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/akash_gemperts/gaussian-splatting"
VENV_PATH="$PROJECT_DIR/.venv"
VDA_DIR="$PROJECT_DIR/Video-Depth-Anything"
VIDEO_CHUNKS_DIR="$PROJECT_DIR/data/pantograph_scene/vda_video_chunks"
OUTPUT_DIR="$PROJECT_DIR/data/pantograph_scene/vda_depth_final"
LOG_DIR="$PROJECT_DIR/data/pantograph_scene/vda_logs"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "VDA Depth Processing - Standalone Script"
echo "=========================================${NC}"
echo "Started: $(date)"
echo ""

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Change to VDA directory
cd "$VDA_DIR"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Function to process a single chunk
process_chunk() {
    local chunk_id=$1
    local gpu_id=$2
    
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Processing Chunk $chunk_id on GPU $gpu_id"
    echo "==========================================${NC}"
    echo "Started: $(date)"
    
    local input_video="$VIDEO_CHUNKS_DIR/chunk_${chunk_id}.mp4"
    local output_dir="$OUTPUT_DIR/chunk_${chunk_id}"
    local log_file="$LOG_DIR/chunk_${chunk_id}_standalone.log"
    
    # Check if input video exists
    if [ ! -f "$input_video" ]; then
        echo -e "${RED}ERROR: Input video not found: $input_video${NC}"
        return 1
    fi
    
    # Run VDA processing
    echo "Command: CUDA_VISIBLE_DEVICES=$gpu_id python run.py --encoder vits --input_video $input_video --output_dir $output_dir --save_npz --grayscale --fp32"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$gpu_id python run.py \
        --encoder vits \
        --input_video "$input_video" \
        --output_dir "$output_dir" \
        --save_npz \
        --grayscale \
        --fp32 \
        2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Chunk $chunk_id completed successfully${NC}"
        echo "Finished: $(date)"
        echo ""
        echo "Output files:"
        ls -lh "$output_dir"/*.npz 2>/dev/null || echo -e "${RED}WARNING: No NPZ files found!${NC}"
        ls -lh "$output_dir"/*.mp4 2>/dev/null || echo -e "${YELLOW}Note: No MP4 files found${NC}"
    else
        echo -e "${RED}✗ Chunk $chunk_id failed with exit code: $exit_code${NC}"
        return 1
    fi
    
    echo ""
    return 0
}

# Main processing
echo -e "${YELLOW}Processing mode: Sequential (one chunk at a time)${NC}"
echo "This ensures each chunk completes and saves before moving to the next"
echo ""

# Process chunks sequentially
CHUNKS=("00" "01" "02" "03" "04")
GPUS=(0 1 2 3 4)
FAILED_CHUNKS=()

for i in "${!CHUNKS[@]}"; do
    chunk="${CHUNKS[$i]}"
    gpu="${GPUS[$i]}"
    
    if ! process_chunk "$chunk" "$gpu"; then
        FAILED_CHUNKS+=("$chunk")
    fi
    
    # Small delay between chunks
    sleep 2
done

# Final summary
echo ""
echo -e "${BLUE}=========================================="
echo "VDA Processing Complete"
echo "==========================================${NC}"
echo "Finished: $(date)"
echo ""

if [ ${#FAILED_CHUNKS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All chunks processed successfully!${NC}"
    echo ""
    echo "Output summary:"
    for chunk in "${CHUNKS[@]}"; do
        npz_file="$OUTPUT_DIR/chunk_${chunk}/chunk_${chunk}_depths.npz"
        if [ -f "$npz_file" ]; then
            size=$(ls -lh "$npz_file" | awk '{print $5}')
            echo -e "  ${GREEN}✓${NC} Chunk $chunk: $size"
        else
            echo -e "  ${RED}✗${NC} Chunk $chunk: NPZ file not found"
        fi
    done
else
    echo -e "${RED}✗ Some chunks failed:${NC}"
    for chunk in "${FAILED_CHUNKS[@]}"; do
        echo "  - Chunk $chunk"
    done
    echo ""
    echo "Check logs in: $LOG_DIR"
    exit 1
fi

echo ""
echo -e "${GREEN}Success! Ready to merge depth chunks.${NC}"
