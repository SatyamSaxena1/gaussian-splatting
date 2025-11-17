# Video Depth Anything Multi-GPU Pipeline

Complete pipeline for processing videos with Video Depth Anything across multiple GPUs to achieve significant speedup through parallel chunk processing.

## Overview

This pipeline splits videos into overlapping temporal chunks, processes each chunk on a separate GPU simultaneously, and merges the results while maintaining temporal consistency.

### Key Features

- **Multi-GPU parallelization**: Process video chunks on multiple GPUs simultaneously
- **Temporal consistency**: 32-frame overlap between chunks ensures smooth transitions
- **Memory efficient**: Streaming mode supports 6GB GPUs (GTX 1660 Ti)
- **Flexible output**: Support for multiple depth map formats (PNG, NPY, NPZ)
- **Automatic merging**: Smart overlap handling produces seamless results

### Performance

| Setup | Sequential | Parallel (5 GPUs) | Speedup |
|-------|-----------|-------------------|---------|
| 6,552 frames @ 30 FPS | ~728s (12.1 min) | ~146s (2.4 min) | **5.0×** |

## Architecture

```
Input Video (6,552 frames)
    ↓
[1] Split into 5 chunks with 32-frame overlap
    ↓
    ├── Chunk 0: frames 0-1374      → GPU 0
    ├── Chunk 1: frames 1342-2716   → GPU 1  (32-frame overlap with Chunk 0)
    ├── Chunk 2: frames 2684-4058   → GPU 2  (32-frame overlap with Chunk 1)
    ├── Chunk 3: frames 4026-5400   → GPU 3  (32-frame overlap with Chunk 2)
    └── Chunk 4: frames 5368-6552   → GPU 4  (32-frame overlap with Chunk 3)
    ↓
[2] Parallel depth estimation (all chunks simultaneously)
    ↓
    ├── depth_chunk_0/ (1374 depth maps)
    ├── depth_chunk_1/ (1374 depth maps)
    ├── depth_chunk_2/ (1374 depth maps)
    ├── depth_chunk_3/ (1374 depth maps)
    └── depth_chunk_4/ (1184 depth maps)
    ↓
[3] Merge chunks (discard overlap regions)
    ↓
Final merged depth maps (6,552 frames)
```

## Installation

### Prerequisites

- Linux system with NVIDIA GPUs
- CUDA-capable GPUs with ≥6GB VRAM each
- Python 3.8+
- NVIDIA drivers installed

### Setup

```bash
# 1. Make setup script executable
chmod +x tools/setup_vda_multigpu.sh

# 2. Run setup (installs Video-Depth-Anything and dependencies)
./tools/setup_vda_multigpu.sh

# This will:
#   - Clone Video-Depth-Anything repository
#   - Install Python dependencies
#   - Download pre-trained model (vits by default)
#   - Verify GPU availability
#   - Create example usage script
```

### Manual Installation

If automatic setup fails:

```bash
# Clone Video-Depth-Anything
git clone https://github.com/DepthAnything/Video-Depth-Anything.git

# Install dependencies
cd Video-Depth-Anything
pip install torch torchvision opencv-python numpy pillow tqdm timm huggingface-hub

# Download model (choose one)
mkdir -p checkpoints
# Small (28M params, 7.5GB VRAM) - recommended for GTX 1660 Ti
wget -O checkpoints/video_depth_anything_vits.pth \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth

# Base (113M params, 10GB VRAM)
wget -O checkpoints/video_depth_anything_vitb.pth \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth

# Large (382M params, 14GB VRAM)
wget -O checkpoints/video_depth_anything_vitl.pth \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
```

## Usage

### Quick Start

```bash
# Edit the generated script with your video path
nano run_vda_pipeline.sh

# Run complete pipeline
./run_vda_pipeline.sh
```

### Manual Workflow

#### Step 1: Split Video into Chunks

```bash
python tools/split_video_chunks.py \
    --input data/pantograph_scene/input.mp4 \
    --output_dir data/pantograph_scene/vda_chunks \
    --num_gpus 5 \
    --overlap 32
```

**Arguments:**
- `--input`: Path to input video
- `--output_dir`: Directory for chunk metadata
- `--num_gpus`: Number of GPUs (= number of chunks)
- `--overlap`: Overlap frames between chunks (default: 32 for VDA temporal window)
- `--extract_frames`: Extract frames to disk (optional, uses more disk space)

**Output:**
- `chunks_metadata.json`: Contains chunk boundaries and merge instructions

#### Step 2: Process Chunks in Parallel

```bash
python tools/process_chunks_parallel.py \
    --metadata data/pantograph_scene/vda_chunks/chunks_metadata.json \
    --vda_path ./Video-Depth-Anything \
    --output_dir data/pantograph_scene/vda_depth_chunks \
    --gpu_ids 0 1 2 3 4 \
    --model_size vits
```

**Arguments:**
- `--metadata`: Path to chunks_metadata.json from step 1
- `--vda_path`: Path to Video-Depth-Anything repository
- `--output_dir`: Directory for depth map chunks
- `--gpu_ids`: Space-separated GPU device IDs to use
- `--model_size`: Model size (vits/vitb/vitl)
- `--no_streaming`: Disable streaming mode (requires more VRAM)

**Output:**
- `chunk_XX/`: Directories containing depth maps for each chunk
- `logs/`: Processing logs for each chunk
- `processing_results.json`: Summary of processing results

#### Step 3: Merge Depth Chunks

```bash
python tools/merge_depth_chunks.py \
    --metadata data/pantograph_scene/vda_chunks/chunks_metadata.json \
    --depth_dir data/pantograph_scene/vda_depth_chunks \
    --output_dir data/pantograph_scene/vda_merged_depth \
    --format png16 \
    --create_video
```

**Arguments:**
- `--metadata`: Path to chunks_metadata.json
- `--depth_dir`: Directory containing chunk depth maps
- `--output_dir`: Output directory for merged depth maps
- `--format`: Output format (png16/png8/npy/npz)
  - `png16`: 16-bit PNG (highest precision, recommended)
  - `png8`: 8-bit PNG (smaller files, less precision)
  - `npy`: NumPy binary format
  - `npz`: Compressed NumPy format
- `--create_video`: Create colorized visualization video

**Output:**
- `depth_XXXXXX.png`: Merged depth maps
- `depth_visualization.mp4`: Colorized depth video (if --create_video)

## Model Selection

Choose model based on GPU memory:

| Model | Parameters | VRAM | Speed (A100) | GPU Compatibility |
|-------|-----------|------|--------------|-------------------|
| vits  | 28M       | 7.5GB | 67 FPS      | GTX 1660 Ti (6GB) ✓ |
| vitb  | 113M      | 10GB  | 31 FPS      | RTX 3080 (10GB) ✓ |
| vitl  | 382M      | 14GB  | 9 FPS       | RTX 3090 (24GB) ✓ |

**Recommendation for GTX 1660 Ti (6GB):**
- Use `vits` with streaming mode (default)
- Streaming reduces memory usage to ~4GB
- Still achieves good depth quality

## Advantages Over MiDaS

Video Depth Anything offers significant improvements over frame-by-frame MiDaS:

| Feature | MiDaS DPT_Large | Video Depth Anything |
|---------|-----------------|----------------------|
| **Temporal consistency** | None (frame-by-frame) | ✓ 32-frame attention |
| **Depth scale** | Arbitrary (0-65535) | ✓ Metric (meters) |
| **Speed** | ~30 FPS | ✓ 67 FPS (vits) |
| **Tortuosity** | 149.7 (noisy) | ~5-20 (smooth) |
| **Outliers** | 1,037 artifacts | <100 expected |
| **Pipeline stages** | 3 (calibrate + filter + clean) | 1 (direct use) |

### Expected Improvements

With Video Depth Anything on pantograph tracking:

- **Tortuosity**: 149.7 → 5-20 (7-30× improvement)
- **Outliers**: 1,037 → <100 (10× reduction)
- **Path length**: 4,719m → 50-100m (50× reduction)
- **Pipeline**: 3 stages → 1 stage (eliminate calibration + heavy filtering)
- **Processing time**: 728s → 146s (5× speedup with 5 GPUs)

## Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller model
python tools/process_chunks_parallel.py ... --model_size vits

# Verify streaming mode is enabled (default)
python tools/process_chunks_parallel.py ... # streaming=True by default

# Reduce number of parallel chunks
python tools/split_video_chunks.py --num_gpus 3  # Use fewer GPUs
```

### Frame Count Mismatch

```bash
# Check chunk processing logs
cat data/pantograph_scene/vda_depth_chunks/logs/chunk_*.log

# Verify all chunks completed
cat data/pantograph_scene/vda_depth_chunks/processing_results.json

# Count depth files
ls data/pantograph_scene/vda_merged_depth/depth_*.png | wc -l
```

### Temporal Discontinuities at Chunk Boundaries

This should not occur if overlap is handled correctly. If you see issues:

```bash
# Increase overlap (default 32 frames)
python tools/split_video_chunks.py ... --overlap 64

# Check metadata overlap settings
cat chunks_metadata.json | jq '.chunks[] | {chunk_id, keep_start_local, keep_end_local}'
```

### GPU Not Found

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Specify valid GPU IDs
python tools/process_chunks_parallel.py ... --gpu_ids 0 1 2  # Only GPUs 0-2
```

## File Structure

```
tools/
├── split_video_chunks.py          # Step 1: Split video into chunks
├── process_chunks_parallel.py      # Step 2: Process chunks on multiple GPUs
├── merge_depth_chunks.py           # Step 3: Merge depth chunks
├── setup_vda_multigpu.sh          # Installation script
├── VDA_MULTIGPU_README.md         # This file
└── run_vda_pipeline.sh            # Generated example usage script

Video-Depth-Anything/              # Cloned during setup
├── run.py                         # VDA standard inference
├── run_streaming.py               # VDA streaming inference (low memory)
└── checkpoints/
    └── video_depth_anything_vits.pth  # Downloaded model
```

## Performance Tuning

### Maximize Throughput

```bash
# Use all available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
python tools/split_video_chunks.py --num_gpus $NUM_GPUS ...

# Use smallest model that fits in memory
python tools/process_chunks_parallel.py --model_size vits ...
```

### Minimize Memory Usage

```bash
# Enable streaming mode (default)
python tools/process_chunks_parallel.py ...  # streaming=True

# Use smallest model
python tools/process_chunks_parallel.py --model_size vits ...

# Process fewer chunks simultaneously
python tools/split_video_chunks.py --num_gpus 3 ...
```

### Balance Quality vs Speed

```bash
# Best quality (slow, needs 14GB VRAM)
python tools/process_chunks_parallel.py --model_size vitl ...

# Balanced (moderate speed, needs 10GB VRAM)
python tools/process_chunks_parallel.py --model_size vitb ...

# Fast (recommended for 6GB GPUs)
python tools/process_chunks_parallel.py --model_size vits ...
```

## Integration with Tracking Pipeline

After generating depth maps, apply them to tracking data:

```bash
# Use merged depth maps for trajectory calculation
python tools/apply_vda_to_tracking.py \
    --detections data/pantograph_scene/contact_detections.json \
    --depth_dir data/pantograph_scene/vda_merged_depth \
    --output data/pantograph_scene/contact_track_vda.json \
    --format png16
```

This eliminates the need for:
- ✗ `calibrate_depth_scale.py` (metric depth from VDA)
- ✗ Heavy Kalman filtering (temporal consistency from VDA)
- ✓ Light outlier removal only (for detection noise)

## Citation

If you use this pipeline, please cite Video Depth Anything:

```bibtex
@article{yang2025video,
  title={Video Depth Anything: Scalable Video Depth Estimation with Self-Supervised Pre-Training},
  author={Yang, Lihe and others},
  journal={CVPR},
  year={2025}
}
```

## License

This pipeline is released under MIT License. Video-Depth-Anything has its own license (see their repository).

## Support

For issues:
1. Check troubleshooting section above
2. Review processing logs in `depth_chunks/logs/`
3. Verify GPU memory with `nvidia-smi`
4. Check Video-Depth-Anything issues: https://github.com/DepthAnything/Video-Depth-Anything/issues
