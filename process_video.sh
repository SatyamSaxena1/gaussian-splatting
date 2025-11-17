#!/bin/bash

# Gaussian Splatting Video Processing Script
# This script extracts frames from a video and prepares them for COLMAP processing

VIDEO_PATH="$1"
OUTPUT_DIR="$2"
FPS="${3:-2}"  # Default: extract 2 frames per second
QUALITY="${4:-2}"  # Default: scale to 1/2 resolution (recommended for faster processing)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Gaussian Splatting Video Processor"
echo "=========================================="
echo ""

# Check if video path is provided
if [ -z "$VIDEO_PATH" ]; then
    echo -e "${RED}Error: No video path provided${NC}"
    echo "Usage: $0 <video_path> <output_dir> [fps] [quality]"
    echo ""
    echo "Arguments:"
    echo "  video_path  - Path to input video file"
    echo "  output_dir  - Output directory for processed data"
    echo "  fps         - Frames per second to extract (default: 2)"
    echo "  quality     - Quality/scale factor:"
    echo "                1 = original, 2 = 1/2 scale, 4 = 1/4 scale (default: 2)"
    echo ""
    echo "Example:"
    echo "  $0 ~/Downloads/video.mp4 ./data/my_scene 2 2"
    exit 1
fi

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo -e "${RED}Error: Video file not found: $VIDEO_PATH${NC}"
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed${NC}"
    echo "Install with: sudo apt-get install ffmpeg"
    exit 1
fi

# Create output directory structure
mkdir -p "$OUTPUT_DIR/input"

echo -e "${GREEN}Video Information:${NC}"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration,nb_frames -of default=noprint_wrappers=1 "$VIDEO_PATH" 2>&1 | grep -E "width|height|r_frame_rate|duration"
echo ""

# Get video duration
DURATION=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO_PATH")
ESTIMATED_FRAMES=$(echo "$DURATION * $FPS" | bc | cut -d. -f1)

echo -e "${YELLOW}Extraction Settings:${NC}"
echo "  FPS: $FPS frames/second"
echo "  Quality: 1/$QUALITY scale"
echo "  Estimated frames: ~$ESTIMATED_FRAMES"
echo ""

read -p "Continue with extraction? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${GREEN}Step 1/3: Extracting frames from video...${NC}"
echo "This may take several minutes..."

# Extract frames with scaling
if [ "$QUALITY" -eq 1 ]; then
    # Original resolution
    ffmpeg -i "$VIDEO_PATH" -vf "fps=$FPS" -qscale:v 2 "$OUTPUT_DIR/input/frame_%04d.jpg" -hide_banner -loglevel error -stats
else
    # Scaled resolution
    ffmpeg -i "$VIDEO_PATH" -vf "fps=$FPS,scale=iw/$QUALITY:ih/$QUALITY" -qscale:v 2 "$OUTPUT_DIR/input/frame_%04d.jpg" -hide_banner -loglevel error -stats
fi

FRAME_COUNT=$(ls -1 "$OUTPUT_DIR/input"/*.jpg 2>/dev/null | wc -l)
echo ""
echo -e "${GREEN}✓ Extracted $FRAME_COUNT frames${NC}"
echo ""

echo -e "${GREEN}Step 2/3: Running COLMAP for camera poses...${NC}"
echo "This will take a while (possibly 10-30 minutes depending on frame count)..."
echo ""

# Check if COLMAP is installed
if ! command -v colmap &> /dev/null; then
    echo -e "${YELLOW}Warning: COLMAP is not installed${NC}"
    echo "You need to install COLMAP to continue:"
    echo "  sudo apt-get install colmap"
    echo ""
    echo -e "Frames saved to: ${GREEN}$OUTPUT_DIR/input/${NC}"
    echo "After installing COLMAP, run:"
    echo "  python convert.py -s $OUTPUT_DIR"
    exit 0
fi

# Run COLMAP through convert.py
cd /home/akash_gemperts/gaussian-splatting
python convert.py -s "$OUTPUT_DIR" --skip_matching

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Processing Complete!${NC}"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Frames extracted: $FRAME_COUNT"
echo ""
echo -e "${GREEN}Step 3/3: Train Gaussian Splatting model${NC}"
echo "To start training, run:"
echo ""
echo "  cd /home/akash_gemperts/gaussian-splatting"
echo "  conda activate gaussian_splatting"
echo "  python train.py -s $OUTPUT_DIR"
echo ""
echo "For lower memory usage, add: --data_device cpu"
echo "For faster training, add: --optimizer_type sparse_adam"
echo ""
