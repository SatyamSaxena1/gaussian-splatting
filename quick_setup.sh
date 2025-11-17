#!/bin/bash

echo "=========================================="
echo "Gaussian Splatting Quick Setup Script"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Installing Miniconda..."
    cd ~
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    
    # Initialize conda
    ~/miniconda3/bin/conda init bash
    
    echo "✅ Miniconda installed!"
    echo "⚠️  Please run 'source ~/.bashrc' and then run this script again"
    exit 0
else
    echo "✅ Conda found: $(conda --version)"
fi

# Navigate to repo
cd /home/akash_gemperts/gaussian-splatting

# Check if environment exists
if conda env list | grep -q "gaussian_splatting"; then
    echo "⚠️  Environment 'gaussian_splatting' already exists"
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n gaussian_splatting -y
    else
        echo "Skipping environment creation"
        exit 0
    fi
fi

echo ""
echo "Creating conda environment (this may take several minutes)..."
echo ""

# Create environment with CUDA 12 compatible setup
conda create -n gaussian_splatting python=3.9 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate gaussian_splatting

# Install PyTorch with CUDA 12 support
echo "Installing PyTorch with CUDA 12 support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
echo "Installing other dependencies..."
conda install -c conda-forge plyfile tqdm -y
pip install opencv-python joblib

# Set CUDA paths
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

# Install custom CUDA extensions
echo "Installing custom CUDA extensions..."
echo "  - diff-gaussian-rasterization..."
pip install submodules/diff-gaussian-rasterization

echo "  - simple-knn..."
pip install submodules/simple-knn

echo "  - fused-ssim..."
pip install submodules/fused-ssim

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To verify installation, run:"
echo "  conda activate gaussian_splatting"
echo "  python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
echo ""
echo "To start training, use:"
echo "  python train.py -s <path_to_dataset>"
echo ""
echo "For more information, see SETUP_INSTRUCTIONS.md"
echo ""
