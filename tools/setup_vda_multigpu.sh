#!/bin/bash
# Setup script for Video Depth Anything multi-GPU pipeline
# This script clones Video-Depth-Anything and sets up the environment

set -e  # Exit on error

echo "============================================================"
echo "Video Depth Anything Multi-GPU Setup"
echo "============================================================"

# Configuration
VDA_REPO="https://github.com/DepthAnything/Video-Depth-Anything.git"
VDA_DIR="./Video-Depth-Anything"
MODEL_SIZE="vits"  # Options: vits (28M), vitb (113M), vitl (382M)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi not found. NVIDIA drivers may not be installed.${NC}"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ Found ${NUM_GPUS} GPU(s)${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Clone Video-Depth-Anything if not exists
echo -e "\n${YELLOW}Setting up Video-Depth-Anything...${NC}"
if [ -d "$VDA_DIR" ]; then
    echo -e "${YELLOW}Directory $VDA_DIR already exists. Pulling latest changes...${NC}"
    cd "$VDA_DIR"
    git pull
    cd ..
else
    echo "Cloning Video-Depth-Anything repository..."
    git clone "$VDA_REPO" "$VDA_DIR"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ Virtual environment not found at .venv${NC}"
    echo "Please activate your existing environment or create one:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    exit 1
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install Video-Depth-Anything dependencies
echo -e "\n${YELLOW}Installing Video-Depth-Anything dependencies...${NC}"
cd "$VDA_DIR"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}Creating requirements.txt...${NC}"
    cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision
opencv-python
numpy
pillow
tqdm
timm
huggingface-hub
EOF
fi

pip install -r requirements.txt

# Download pre-trained models
echo -e "\n${YELLOW}Downloading pre-trained models...${NC}"

# Create checkpoints directory
mkdir -p checkpoints

# Model URLs (from Hugging Face)
VITS_URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
VITB_URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth"
VITL_URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"

download_model() {
    local model=$1
    local url=$2
    local file="checkpoints/video_depth_anything_${model}.pth"
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ Model ${model} already downloaded${NC}"
    else
        echo "Downloading ${model} model..."
        wget -O "$file" "$url" || curl -L -o "$file" "$url"
        if [ -f "$file" ]; then
            echo -e "${GREEN}✓ Downloaded ${model} model${NC}"
        else
            echo -e "${RED}✗ Failed to download ${model} model${NC}"
            return 1
        fi
    fi
}

# Download requested model (default: vits for 6GB GPUs)
if [ "$MODEL_SIZE" == "vits" ]; then
    download_model "vits" "$VITS_URL"
elif [ "$MODEL_SIZE" == "vitb" ]; then
    download_model "vitb" "$VITB_URL"
elif [ "$MODEL_SIZE" == "vitl" ]; then
    download_model "vitl" "$VITL_URL"
else
    echo -e "${YELLOW}Unknown model size: $MODEL_SIZE. Downloading vits (small) by default.${NC}"
    download_model "vits" "$VITS_URL"
fi

cd ..

# Verify tools exist
echo -e "\n${YELLOW}Verifying pipeline tools...${NC}"
TOOLS_DIR="./tools"
REQUIRED_TOOLS=(
    "split_video_chunks.py"
    "process_chunks_parallel.py"
    "merge_depth_chunks.py"
)

for tool in "${REQUIRED_TOOLS[@]}"; do
    if [ -f "$TOOLS_DIR/$tool" ]; then
        echo -e "${GREEN}✓ $tool${NC}"
    else
        echo -e "${RED}✗ $tool not found${NC}"
    fi
done

# Test imports
echo -e "\n${YELLOW}Testing Python dependencies...${NC}"
python -c "
import torch
import cv2
import numpy as np
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPUs: {torch.cuda.device_count()}')
print(f'✓ OpenCV {cv2.__version__}')
print(f'✓ NumPy {np.__version__}')
"

# Create example usage script
echo -e "\n${YELLOW}Creating example usage script...${NC}"
cat > run_vda_pipeline.sh << 'EOF'
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
EOF

chmod +x run_vda_pipeline.sh

# Print summary
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo -e "\nVideo-Depth-Anything installed at: ${GREEN}$VDA_DIR${NC}"
echo -e "Model: ${GREEN}$MODEL_SIZE${NC}"
echo -e "Available GPUs: ${GREEN}$NUM_GPUS${NC}"
echo ""
echo -e "${YELLOW}Memory requirements per GPU:${NC}"
echo "  vits (Small):  ~7.5 GB VRAM (recommended for GTX 1660 Ti)"
echo "  vitb (Base):   ~10 GB VRAM"
echo "  vitl (Large):  ~14 GB VRAM"
echo ""
echo -e "${YELLOW}Usage:${NC}"
echo "  1. Edit run_vda_pipeline.sh with your video path"
echo "  2. Run: ./run_vda_pipeline.sh"
echo ""
echo -e "${YELLOW}Manual usage:${NC}"
echo "  # Split video"
echo "  python tools/split_video_chunks.py --input video.mp4 --output_dir chunks/ --num_gpus 5"
echo ""
echo "  # Process chunks"
echo "  python tools/process_chunks_parallel.py --metadata chunks/chunks_metadata.json --vda_path $VDA_DIR --output_dir depth/"
echo ""
echo "  # Merge chunks"
echo "  python tools/merge_depth_chunks.py --metadata chunks/chunks_metadata.json --depth_dir depth/ --output_dir merged/"
echo ""
echo -e "${GREEN}Ready to process videos!${NC}"
