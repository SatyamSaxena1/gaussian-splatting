# Railway Pantograph 3D Reconstruction System

A specialized 3D reconstruction pipeline for railway pantograph contact point tracking using Gaussian Splatting with Video Depth Anything (VDA) integration.

## Project Overview

This project extends the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) framework to solve a specific railway infrastructure problem: **tracking pantograph contact points on overhead catenary systems from moving train-mounted cameras**.

**Real-world application**: Railway operators need to monitor where pantographs (the equipment that collects power from overhead wires) make contact with catenary systems to prevent wear, predict maintenance, and ensure reliable power collection.

### The Challenge

Traditional visual inspection methods struggle with:
- **High-speed motion blur** from moving trains (30+ FPS video)
- **Complex 3D geometry** of overhead catenary infrastructure
- **Dynamic vs static separation** - pantograph moves while scenery is static
- **Precise 3D positioning** needed for wear analysis
- **Temporal consistency** - tracking same contact point across frames

### Our Solution

Through iterative development over 117 development sessions, we built a complete pipeline:

1. **Video preprocessing** - Frame extraction from train-mounted camera footage
2. **Motion segmentation** - Separate semi-static pantograph from static infrastructure  
3. **Temporally-consistent depth** - Video Depth Anything (VDA) for smooth depth maps
4. **Robust tracking** - YOLO11 detection + Kalman filtering for smooth trajectories
5. **3D reconstruction** - Gaussian Splatting for photorealistic static scene models
6. **Industry visualization** - USD export for analysis in tools like Pixar's USDView

**Key innovation**: Multi-GPU video processing with crash-resistant design, achieving 5× speedup while handling long-running tasks that caused IDE crashes.

## Key Features

### 🎥 Video Processing Pipeline
- Frame extraction and preprocessing
- Motion-based segmentation (static infrastructure vs. dynamic pantograph)
- Multi-GPU depth estimation (5× speedup)
- Temporal consistency across frames

### 🎯 Contact Point Tracking
- YOLO11-based pantograph detection
- Kalman filtering for smooth trajectories
- 3D position estimation using depth maps
- Outlier removal and interpolation

### 🏗️ 3D Reconstruction
- Static scene reconstruction with Gaussian Splatting
- Dynamic object tracking overlaid on static model
- Pole detection using RANSAC
- USD export for visualization in industry tools (e.g., USD View)

### ⚡ Multi-GPU Optimization
- Video chunking for parallel processing
- 5× faster depth estimation on 5× GTX 1660 Ti GPUs
- Crash-resistant processing with automatic recovery
- Memory-efficient handling of long videos

## Technical Stack

### Core Technologies
- **3D Gaussian Splatting**: Scene reconstruction
- **Video Depth Anything (VDA)**: Temporally-consistent monocular depth estimation
- **YOLO11**: Object detection for pantograph tracking
- **COLMAP**: Structure-from-Motion for camera pose estimation
- **PyTorch**: Deep learning framework
- **USD (Universal Scene Description)**: 3D data exchange format

### Hardware Requirements
- NVIDIA GPU with 6GB+ VRAM (tested on GTX 1660 Ti)
- Multi-GPU setup supported (linear scaling)
- 24GB+ RAM recommended for video processing

## Installation

```bash
# Clone repository
git clone https://github.com/SatyamSaxena1/gaussian-splatting.git
cd gaussian-splatting

# Initialize submodules
git submodule update --init --recursive

# Install dependencies (creates conda environment)
./quick_setup.sh

# Install COLMAP
sudo apt-get install colmap
```

## Quick Start

### 1. Process Your Video

```bash
# Extract frames and run preprocessing
./process_video.sh path/to/video.mp4 data/my_scene 2 2

# Generate depth maps (multi-GPU)
./run_vda_standalone.sh

# Run COLMAP for camera poses
python convert.py -s data/my_scene
```

### 2. Track Pantograph Contact Points

```bash
# Detect and track contact points
python tools/track_contact_yolo.py \
    --video data/my_scene/input.mp4 \
    --output data/my_scene/contact_track.json

# Enrich with 3D positions from depth
python tools/apply_vda_depth_simple.py \
    --tracking data/my_scene/contact_track.json \
    --depth_dir data/my_scene/vda_merged_depth \
    --output data/my_scene/contact_track_3d.json
```

### 3. Train Gaussian Splatting Model

```bash
conda activate gaussian_splatting

# Train on static infrastructure
python train.py -s data/my_scene --data_device cpu
```

### 4. Export and Visualize

```bash
# Export tracking data to USD
python tools/export_contact_track_to_usd.py \
    --tracking data/my_scene/contact_track_3d.json \
    --output data/my_scene/contact_track.usda

# View in USD View
usdview data/my_scene/contact_track.usda
```

## Project Structure

```
gaussian-splatting/
├── tools/                          # Custom processing tools
│   ├── split_video_chunks.py       # Video chunking for multi-GPU
│   ├── merge_depth_chunks.py       # Depth map merging
│   ├── track_contact_yolo.py       # Pantograph detection
│   ├── apply_vda_depth_simple.py   # 3D position estimation
│   ├── kalman_filter_trajectory.py # Trajectory smoothing
│   ├── export_*_to_usd.py          # USD export tools
│   └── ...
├── pole_detection/                 # Pole detection module (RANSAC)
├── docs/                           # Additional documentation
│   ├── PANTOGRAPH_PIPELINE.md      # Pipeline overview
│   ├── RUN_NEW_VIDEO.md            # Video processing guide
│   └── USDVIEW_SETUP.md            # USD visualization setup
├── run_vda_standalone.sh           # Multi-GPU depth processing
├── process_video.sh                # Video preprocessing
└── quick_setup.sh                  # Environment setup
```

## Documentation

- **[START_HERE.md](START_HERE.md)** - Quick start guide
- **[VDA_IMPLEMENTATION_SUMMARY.md](VDA_IMPLEMENTATION_SUMMARY.md)** - Multi-GPU depth processing details
- **[VIDEO_PROCESSING_GUIDE.md](VIDEO_PROCESSING_GUIDE.md)** - Complete video processing workflow
- **[docs/PANTOGRAPH_PIPELINE.md](docs/PANTOGRAPH_PIPELINE.md)** - Full pipeline explanation
- **[docs/RUN_NEW_VIDEO.md](docs/RUN_NEW_VIDEO.md)** - Step-by-step video processing
- **[VDA_STANDALONE_GUIDE.md](VDA_STANDALONE_GUIDE.md)** - Multi-GPU setup guide

## Key Innovations

### 1. Multi-GPU Video Depth Processing
**Problem**: Single GPU took ~3 hours for 6,500 frames. VSCode/IDE crashes killed background processes.

**Solution**: 
- Video chunking with 32-frame overlap for temporal consistency
- Sequential processing across 5 GPUs (not parallel to avoid crashes)
- Crash-resistant standalone scripts (survive IDE/system crashes)
- **Result**: 35 minutes total (5× speedup), 99% track enrichment success

### 2. Temporal Depth Consistency
**Problem**: Single-frame depth methods (MiDaS) caused jitter in tracking.

**Solution**:
- Switched to Video Depth Anything (temporal CNN model)
- Maintains smooth transitions between frames
- Better for tracking moving objects
- **Trade-off**: Requires FP32 on GTX 1660 Ti (no Tensor Cores for FP16)

### 3. Static/Dynamic Separation
**Problem**: Pantograph moves while infrastructure is static - standard reconstruction fails.

**Solution**:
- Motion-based segmentation identifies pantograph as foreground
- Separate reconstruction: static (Gaussian Splatting) + dynamic (tracking)
- Depth-aware backprojection for 3D contact positions
- **Result**: Accurate 3D positions on reconstructed infrastructure

### 4. Production-Ready Tracking Pipeline
**Evolution**: Optical flow → Manual labeling → YOLO11 → Kalman filtering → Outlier removal

**Components**:
- YOLO11 custom-trained on pantograph contact points
- Kalman filter for smooth trajectories (handles occlusion)
- Outlier detection and cubic interpolation
- USD export for industry-standard visualization tools

**Result**: Smooth, accurate 3D trajectories suitable for wear analysis

## Development Journey

This project evolved through **117 development sessions** solving real engineering challenges:

### Phase 1: Foundation (Sessions 1-20)
- Integrated USD visualization for development feedback
- Set up Gaussian Splatting pipeline for railway videos
- Established USD export workflow

### Phase 2: Detection & Segmentation (Sessions 20-50)
- Identified pantograph separation challenge
- Built depth-based segmentation pipeline
- Migrated from optical flow to YOLOv8 (more robust)
- Upgraded YOLOv8 → YOLO11 (Session 55, using Pantograph.v1i dataset)
- Manual labeling and training data curation with LabelImg
- Debugged bounding box failures

### Phase 3: 3D Tracking (Sessions 50-70)
- Implemented contact point tracking with bounding boxes
- Added trajectory analysis (position, velocity, dynamics)
- Integrated MiDaS depth for initial Z-axis estimation
- Rendered depth visualization videos

### Phase 4: Video Depth Anything (Sessions 70-90)
- Migrated MiDaS → VDA for temporal consistency
- Investigated multi-GPU capability
- Debugged FP32/FP16 issues on GTX 1660 Ti
- Used Context7 MCP to research VDA documentation

### Phase 5: Production Hardening (Sessions 90-117)
- Solved VSCode crash issues killing background jobs
- Redesigned for crash-resistant standalone execution
- Implemented multi-GPU chunked processing
- Integrated COLMAP for accurate camera poses
- Achieved production-ready 5× speedup

**Key Learnings**:
- Long-running GPU tasks need crash isolation from IDE
- Temporal consistency critical for tracking applications
- GTX 1660 Ti lacks Tensor Cores (FP16 autocast → NaN in VDA)
- Sequential multi-GPU more reliable than parallel for long jobs

### Tested Configuration
- **Video**: 6,553 frames @ 30 FPS, 848×478 resolution
- **GPUs**: 5× NVIDIA GeForce GTX 1660 Ti (6GB each)
- **Processing Time**: ~35 minutes (depth estimation)
- **Output**: 6,681 temporally-consistent depth frames
- **Tracking Success**: 99.0% (6,484/6,552 tracks enriched)

### Speedup Metrics
- **Multi-GPU**: 5× linear scaling
- **Depth Generation**: 35 min (vs 3 hours single GPU)
- **Memory Usage**: 4.5-4.6 GB VRAM per GPU
- **Total Pipeline**: ~2-3 hours (extraction → depth → tracking → training)

## Credits

### Original Work
This project builds upon:
- **3D Gaussian Splatting** by Kerbl et al. (SIGGRAPH 2023)
  - Paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
  - Original repo: https://github.com/graphdeco-inria/gaussian-splatting

### Dependencies
- **Depth Anything V2** - Monocular depth estimation
- **YOLO11** (Ultralytics) - Object detection (upgraded from YOLOv8)
- **COLMAP** - Structure-from-Motion
- **PyTorch** - Deep learning framework
- **OpenCV** - Video processing
- **USD (Pixar)** - 3D data format

### Our Contributions
- Multi-GPU VDA processing pipeline
- Railway pantograph tracking system
- Static/dynamic scene separation
- 3D contact point estimation
- USD export tools for railway analysis
- Crash-resistant processing scripts
- Comprehensive documentation for railway applications

## License

This project maintains the original Gaussian Splatting license for the core framework. See [LICENSE.md](LICENSE.md).

Our custom tools and modifications (in `tools/`, `pole_detection/`, and associated scripts) are provided as-is for research and railway infrastructure monitoring purposes.

## Citation

If you use this work, please cite both the original Gaussian Splatting paper and acknowledge this railway-specific extension:

```bibtex
@Article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {ACM Transactions on Graphics},
    number       = {4},
    volume       = {42},
    month        = {July},
    year         = {2023},
    url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

@misc{railway-gaussian-splatting,
    title        = {Railway Pantograph 3D Reconstruction System},
    author       = {Saxena, Satyam and Gemperts, Akash},
    year         = {2025},
    url          = {https://github.com/SatyamSaxena1/gaussian-splatting}
}
```

## Contact & Support

For questions about:
- **Railway applications**: Open an issue on this repository
- **Original Gaussian Splatting**: See the [original repository](https://github.com/graphdeco-inria/gaussian-splatting)
- **Video Depth Anything**: See the [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2)

## Acknowledgments

This work was developed for railway infrastructure monitoring applications. We thank:
- The original Gaussian Splatting authors for their groundbreaking work
- The Depth Anything V2 team for temporal depth estimation
- The open-source community for tools like COLMAP, YOLO11, and USD

---

**Status**: Active development for railway pantograph monitoring applications

**Last Updated**: November 2025
