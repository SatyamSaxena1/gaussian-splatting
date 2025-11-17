# Railway Pantograph 3D Reconstruction Pipeline

Complete technical documentation of the processing pipeline from raw video to 3D tracking results.

## Pipeline Overview

```
Raw Video (train-mounted camera)
    ↓
[1. Video Preprocessing] → Frames + Metadata
    ↓
[2. Multi-GPU Depth Estimation] → Depth Maps (VDA)
    ↓
[3. Object Detection & Tracking] → Bounding Boxes (YOLO11)
    ↓
[4. 3D Position Estimation] → 3D Tracks (Depth Fusion)
    ↓
[5. Trajectory Refinement] → Smoothed Tracks (Kalman)
    ↓
[6. 3D Reconstruction] → Static Scene (Gaussian Splatting)
    ↓
[7. Export & Visualization] → USD Files
```

## Stage 1: Video Preprocessing

**Script:** `process_video.sh`

### Purpose
Extract frames from train-mounted camera video at appropriate sampling rate for 3D reconstruction.

### Usage
```bash
./process_video.sh <video_path> <output_dir> [fps] [scale]

# Example:
./process_video.sh ~/Downloads/train_video.mp4 ./data/scene_01 2 2
```

### Parameters
- `video_path`: Input video file (MP4, AVI, etc.)
- `output_dir`: Scene directory for all outputs
- `fps`: Frames per second to extract (default: 2)
  - 1 fps: Slower scenes, lower memory
  - 2 fps: Balanced (recommended)
  - 3-5 fps: High detail, more processing
- `scale`: Resolution scaling factor (default: 2)
  - 1: Original resolution (slow, high quality)
  - 2: Half resolution (recommended)
  - 4: Quarter resolution (fast, lower quality)

### Output Structure
```
data/scene_01/
├── input/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── metadata.json  # Video info
```

### Implementation Details
- Uses FFmpeg for frame extraction
- Maintains aspect ratio during scaling
- JPEG quality: qscale=2 (high quality)
- Validates video existence and codec support

### Performance
- 6,552 frames @ 30 FPS = ~218 seconds source
- Extraction @ 2 FPS = ~437 frames
- Processing time: ~2-5 minutes

---

## Stage 2: Multi-GPU Depth Estimation

**Scripts:** 
- `tools/split_video_chunks.py` - Video chunking
- `tools/extract_video_chunks.py` - Chunk extraction
- `run_vda_standalone.sh` - VDA processing
- `tools/merge_depth_chunks.py` - Depth merging

### Purpose
Generate temporally-consistent depth maps using Video Depth Anything across multiple GPUs.

### Step 2.1: Split Video into Chunks

```bash
python tools/split_video_chunks.py \
    --input data/scene_01/input.mp4 \
    --output_dir data/scene_01/vda_chunks \
    --num_chunks 5 \
    --overlap 32
```

**Parameters:**
- `num_chunks`: Number of GPUs available (5 for 5× GTX 1660 Ti)
- `overlap`: Temporal overlap in frames (32 for VDA temporal window)

**Output:**
```
data/scene_01/vda_chunks/
└── chunks_metadata.json
    {
        "num_chunks": 5,
        "overlap": 32,
        "chunks": [
            {"id": "00", "start_frame": 0, "end_frame": 1342},
            {"id": "01", "start_frame": 1310, "end_frame": 2684},
            ...
        ]
    }
```

### Step 2.2: Extract Video Chunks

```bash
python tools/extract_video_chunks.py \
    --metadata data/scene_01/vda_chunks/chunks_metadata.json \
    --output_dir data/scene_01/vda_video_chunks
```

**Output:**
```
data/scene_01/vda_video_chunks/
├── chunk_00.mp4  (87 MB)
├── chunk_01.mp4  (87 MB)
└── ...
```

### Step 2.3: Run VDA Processing

**Script:** `run_vda_standalone.sh`

**Key Configuration:**
```bash
VENV_PATH="$PROJECT_DIR/.venv"
VDA_DIR="$PROJECT_DIR/Video-Depth-Anything"
VIDEO_CHUNKS_DIR="data/scene_01/vda_video_chunks"
OUTPUT_DIR="data/scene_01/vda_depth_final"
```

**Execution:**
```bash
./run_vda_standalone.sh
```

**Processing Flow:**
```
For each chunk (00, 01, 02, 03, 04):
    1. Activate virtual environment
    2. Set CUDA_VISIBLE_DEVICES=<gpu_id>
    3. Run: python run.py --encoder vits \
                         --input_video chunk_XX.mp4 \
                         --output_dir output/ \
                         --save_npz \
                         --grayscale \
                         --fp32
    4. Save NPZ file immediately (crash-resistant)
    5. Log output to vda_logs/chunk_XX_standalone.log
    6. Wait 2 seconds before next chunk
```

**Critical Flags:**
- `--fp32`: Force FP32 (GTX 1660 Ti lacks Tensor Cores for FP16)
- `--save_npz`: Save compressed depth arrays
- `--grayscale`: Monochrome depth visualization
- `--encoder vits`: Small model (112M, 28M params)

**Output:**
```
data/scene_01/vda_depth_final/
├── chunk_00/chunk_00_depths.npz  (916 MB, 1343 frames)
├── chunk_01/chunk_01_depths.npz  (839 MB, 1375 frames)
└── ...
```

**Performance:**
- Single GPU: ~3 hours for 6,553 frames
- Multi-GPU (5×): ~35 minutes sequential
- Speedup: 5× linear scaling
- VRAM: 4.5-4.6 GB per GPU (75% of 6GB)

### Step 2.4: Merge Depth Chunks

```bash
python tools/merge_depth_chunks.py \
    --metadata data/scene_01/vda_chunks/chunks_metadata.json \
    --depth_dir data/scene_01/vda_depth_final \
    --output_dir data/scene_01/vda_merged_depth \
    --format png16 \
    --create_video
```

**Merging Strategy:**
- Discard overlap regions (32 frames at boundaries)
- Keep non-overlapping sections from each chunk
- Sequential reassembly: chunk_00[0:1310] + chunk_01[1342:2652] + ...

**Output:**
```
data/scene_01/vda_merged_depth/
├── depth_000000.png  (16-bit PNG, ~80 KB each)
├── depth_000001.png
├── ...
├── depth_006680.png
└── depth_visualization.mp4  (61 MB)
```

**Format: PNG16**
- 16-bit grayscale PNG
- Value range: 0-65535
- Normalized depth: depth_value / 65535.0
- Inverse depth representation (closer = higher value)

---

## Stage 3: Object Detection & Tracking

**Scripts:**
- `tools/prepare_yolo_dataset.py` - Dataset preparation
- Manual: LabelImg annotation
- `tools/track_contact_yolo.py` - Contact point tracking

### Step 3.1: Prepare Training Dataset

```bash
python tools/prepare_yolo_dataset.py \
    --frames data/scene_01/input \
    --output data/scene_01/yolo_dataset \
    --sample_rate 30
```

**Sampling Strategy:**
- Extract every Nth frame for manual labeling
- Typical: 180 training + 20 validation samples from 6,500 frames
- Ensure temporal diversity (different pantograph positions)

### Step 3.2: Manual Annotation

**Tool:** LabelImg

```bash
labelimg data/scene_01/yolo_dataset/images/train
```

**Annotation Guidelines:**
- Class: "pantograph" (single class)
- Bounding box: Tight fit around pantograph body
- Contact point heuristic: Top-center of bbox
- Quality over quantity: 200 good annotations > 500 poor

**Output Format (YOLO TXT):**
```
# data/scene_01/yolo_dataset/labels/train/frame_0001.txt
0 0.512 0.345 0.287 0.456
# Format: class_id center_x center_y width height (normalized)
```

### Step 3.3: Train YOLO11 Model

```bash
# Using Ultralytics YOLO11
from ultralytics import YOLO

model = YOLO('yolo11s.pt')  # Start from pretrained
model.train(
    data='data/scene_01/yolo_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

**Training Results (Session 55):**
- Model: YOLO11s (small variant)
- mAP50: 0.852
- Recall: 0.90
- Training time: ~2-3 hours on GTX 1660 Ti

### Step 3.4: Track Contact Points

```bash
python tools/track_contact_yolo.py \
    data/scene_01 \
    models/yolo11s_pantograph.pt \
    --conf-threshold 0.25 \
    --visualize
```

**Processing:**
```python
for frame in video_frames:
    # Run YOLO detection
    results = model(frame, conf=0.25)
    
    for detection in results:
        bbox = detection.xyxy  # [x1, y1, x2, y2]
        confidence = detection.conf
        
        # Contact point = top-center of bbox
        contact_x = (bbox[0] + bbox[2]) / 2
        contact_y = bbox[1]  # Top edge
        
        # Store track
        track = {
            "frame": frame_id,
            "bbox": bbox,
            "confidence": confidence,
            "contact_2d": [contact_x, contact_y]
        }
```

**Output:**
```json
{
    "metadata": {
        "total_frames": 6552,
        "detected_frames": 6448,
        "detection_rate": 0.984
    },
    "tracks": [
        {
            "frame": 0,
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.815,
            "contact_2d": [u, v]
        },
        ...
    ]
}
```

**Performance:**
- Detection rate: 98.4%
- Detection gaps: 21 gaps (mean 5 frames, max 28 frames)
- Average confidence: 0.815

---

## Stage 4: 3D Position Estimation

**Script:** `tools/apply_vda_depth_simple.py`

### Purpose
Convert 2D contact points + depth maps → 3D positions in camera coordinates.

### Usage
```bash
python tools/apply_vda_depth_simple.py \
    --tracking data/scene_01/contact_track_yolo.json \
    --depth_dir data/scene_01/vda_merged_depth \
    --output data/scene_01/contact_track_3d.json
```

### Algorithm

**1. Load depth map for frame:**
```python
depth_file = f"depth_{frame_num:06d}.png"
depth_map = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
depth_map = depth_map.astype(float) / 65535.0  # Normalize
```

**2. Sample depth at contact point:**
```python
# 5×5 window around contact point for robustness
window = depth_map[v-2:v+3, u-2:u+3]
depth = np.median(window[window > 0])  # Ignore zeros
```

**3. Back-project to 3D:**
```python
# Inverse depth representation
z = 1.0 / (depth + 1e-6)

# Camera intrinsics (default or from COLMAP)
fx, fy = 800.0, 800.0  # Focal length
cx, cy = width/2, height/2  # Principal point

# Pinhole camera model
X = (u - cx) * z / fx
Y = (v - cy) * z / fy
Z = z
```

**4. Store 3D position:**
```python
track["vda_position_3d"] = {
    "x": float(X),
    "y": float(Y),
    "z": float(Z),
    "depth_raw": float(depth),
    "depth_std": float(depth_std)
}
```

### Output Format
```json
{
    "tracks": [
        {
            "frame": 0,
            "contact_2d": [512.3, 245.7],
            "vda_position_3d": {
                "x": -2.491,
                "y": -2.949,
                "z": 10.068,
                "depth_raw": 0.0993,
                "depth_std": 0.012
            }
        },
        ...
    ],
    "statistics": {
        "total_tracks": 6552,
        "enriched": 6484,
        "skipped": 68,
        "success_rate": 0.990
    }
}
```

### Statistics (Pantograph Scene)
- Total tracks: 6,552
- Enriched with 3D: 6,484 (99.0%)
- Skipped (no depth): 68 (1.0%)
- X range: [-46.573, 12.038]
- Y range: [-130.467, 0.341]
- Z range: [1.136, 436.709]

---

## Stage 5: Trajectory Refinement

**Script:** `tools/kalman_filter_trajectory.py`

### Purpose
Smooth 3D trajectories using Kalman filtering to handle:
- Detection gaps (104 frames missed)
- Outliers from depth estimation errors
- Sensor noise

### Kalman Filter Configuration

**State Vector (6D):**
```
x = [X, Y, Z, Vx, Vy, Vz]
    Position (3D) + Velocity (3D)
```

**State Transition Matrix (constant velocity model):**
```
F = [[1, 0, 0, dt, 0,  0 ],
     [0, 1, 0, 0,  dt, 0 ],
     [0, 0, 1, 0,  0,  dt],
     [0, 0, 0, 1,  0,  0 ],
     [0, 0, 0, 0,  1,  0 ],
     [0, 0, 0, 0,  0,  1 ]]
```

**Measurement Matrix:**
```
H = [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0]]
```

**Process Noise Covariance:**
```
Q = [[σ_pos², 0, 0, 0, 0, 0],
     [0, σ_pos², 0, 0, 0, 0],
     [0, 0, σ_pos², 0, 0, 0],
     [0, 0, 0, σ_vel², 0, 0],
     [0, 0, 0, 0, σ_vel², 0],
     [0, 0, 0, 0, 0, σ_vel²]]
    
where σ_pos = 0.1, σ_vel = 0.01
```

### Usage
```bash
python tools/kalman_filter_trajectory.py \
    --input data/scene_01/contact_track_3d.json \
    --output data/scene_01/contact_track_smoothed.json \
    --outlier_threshold 3.0
```

### Outlier Detection & Interpolation

**1. Detect outliers:**
```python
# Compute frame-to-frame displacement
displacement = np.linalg.norm(pos[i+1] - pos[i])

# Z-score method
z_score = (displacement - mean) / std
if z_score > threshold:  # threshold = 3.0
    mark_as_outlier()
```

**2. Interpolate missing/outlier frames:**
```python
# Cubic spline interpolation
from scipy.interpolate import CubicSpline

valid_indices = ~outlier_mask
cs_x = CubicSpline(valid_indices, x[valid_indices])
cs_y = CubicSpline(valid_indices, y[valid_indices])
cs_z = CubicSpline(valid_indices, z[valid_indices])

# Fill gaps
x_smooth = cs_x(all_indices)
y_smooth = cs_y(all_indices)
z_smooth = cs_z(all_indices)
```

### Output Statistics
- Input tracks: 6,484
- Outliers detected: 127 (2.0%)
- Gaps filled: 68
- Output tracks: 6,552 (complete)

---

## Stage 6: 3D Reconstruction

**Script:** `train.py` (original Gaussian Splatting)

### Purpose
Reconstruct static scene (infrastructure) using Gaussian Splatting for photorealistic 3D model.

### Prerequisites

**1. COLMAP Camera Poses:**
```bash
python convert.py -s data/scene_01 \
    --colmap_executable colmap \
    --skip_matching
```

**Required COLMAP outputs:**
```
data/scene_01/sparse/0/
├── cameras.bin
├── images.bin
└── points3D.bin
```

### Training

```bash
conda activate gaussian_splatting

python train.py \
    -s data/scene_01 \
    --iterations 30000 \
    --data_device cpu \
    --save_iterations 7000 15000 30000
```

**Parameters:**
- `-s`: Scene directory
- `--iterations`: Training steps (7k=fast, 30k=quality)
- `--data_device cpu`: Offload data to CPU (saves VRAM)
- `--save_iterations`: Checkpoints to save

**Memory Requirements:**
- VRAM: 6-12 GB (depends on point cloud size)
- RAM: 16-32 GB
- Disk: ~500 MB - 2 GB per checkpoint

### Training Output
```
output/<timestamp>/
├── point_cloud/
│   ├── iteration_7000/
│   ├── iteration_15000/
│   └── iteration_30000/
│       └── point_cloud.ply  # 3D Gaussians
├── cameras.json
└── cfg_args
```

### Rendering
```bash
python render.py \
    -m output/<timestamp> \
    --iteration 30000 \
    --skip_train \
    --skip_test
```

**Output:**
```
output/<timestamp>/renders/
├── 00000.png
├── 00001.png
└── ...
```

---

## Stage 7: Export & Visualization

**Scripts:**
- `tools/export_contact_track_to_usd.py`
- `tools/export_gaussians_to_usd.py`
- `tools/open_usdview.sh`

### Purpose
Export tracking data and 3D models to USD format for industry-standard visualization.

### Export Contact Track to USD

```bash
python tools/export_contact_track_to_usd.py \
    --tracking data/scene_01/contact_track_smoothed.json \
    --output data/scene_01/contact_track.usda \
    --fps 30 \
    --point_radius 0.05
```

**USD Structure:**
```
Stage
├── /World
│   └── /ContactTrack
│       ├── frame_0000 (Sphere)
│       ├── frame_0001 (Sphere)
│       └── ...
└── TimeCode (0 to num_frames)
```

**Features:**
- Time-varying sphere positions
- Color-coded by velocity
- Playback at original FPS
- Camera path from COLMAP

### Export Gaussian Splat to USD

```bash
python tools/export_gaussians_to_usd.py \
    --model output/<timestamp>/point_cloud/iteration_30000/point_cloud.ply \
    --output data/scene_01/gaussian_splat.usda \
    --max_points 1000000
```

### Visualize in USDView

```bash
tools/open_usdview.sh data/scene_01/contact_track.usda
```

**USDView Controls:**
- Play/Pause: Spacebar
- Frame navigation: Arrow keys
- Camera: Alt + Mouse drag
- Timeline: Bottom scrubber

---

## Pipeline Validation

### Quality Checks

**1. Frame Extraction:**
```bash
# Check frame count
ls data/scene_01/input/*.jpg | wc -l

# Verify resolution
identify data/scene_01/input/frame_0001.jpg
```

**2. Depth Maps:**
```bash
# Check depth range
python -c "
import cv2
import numpy as np
depth = cv2.imread('data/scene_01/vda_merged_depth/depth_000000.png', -1)
print(f'Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean()}')
"
```

**3. Tracking:**
```bash
# Parse JSON stats
python -c "
import json
with open('data/scene_01/contact_track_3d.json') as f:
    data = json.load(f)
print(data['statistics'])
"
```

**4. 3D Positions:**
```bash
# Visualize trajectory
python tools/analyze_3d_trajectories.py \
    --input data/scene_01/contact_track_smoothed.json \
    --output data/scene_01/trajectory_analysis.png
```

---

## Troubleshooting

### VSCode Crashes During VDA

**Problem:** IDE kills background processes  
**Solution:** Use standalone script
```bash
./run_vda_standalone.sh
# Runs completely independently of IDE
```

### FP16 NaN Outputs

**Problem:** GTX 1660 Ti produces all NaN with FP16  
**Root Cause:** Lacks Tensor Cores for FP16 autocast  
**Solution:** Force FP32 in `run_vda_standalone.sh`
```bash
python run.py --fp32 ...
```

### COLMAP Feature Matching Fails

**Problem:** Too few features extracted  
**Symptoms:** `images.bin` has <50% of input frames  
**Solutions:**
```bash
# Increase SIFT features
python convert.py -s data/scene_01 \
    --sift_extraction.max_num_features 16384

# Use sequential matching
python convert.py -s data/scene_01 \
    --matching sequential \
    --sequential_overlap 5
```

### YOLO Detection Gaps

**Problem:** Pantograph not detected in some frames  
**Causes:** Motion blur, occlusion, lighting changes  
**Solutions:**
- Lower confidence threshold: `--conf-threshold 0.20`
- Add more training samples with diverse conditions
- Use temporal interpolation in Kalman filter

### Out of Memory During Training

**Problem:** GPU OOM during Gaussian Splatting  
**Solutions:**
```bash
# Offload data to CPU
python train.py -s data/scene_01 --data_device cpu

# Reduce point cloud
python convert.py -s data/scene_01 --num_points 100000

# Train at lower resolution
python train.py -s data/scene_01 --resolution 2
```

---

## Performance Benchmarks

### Test Configuration
- Hardware: 5× NVIDIA GTX 1660 Ti (6GB each), 24GB RAM
- Video: 6,553 frames @ 30 FPS, 848×478 resolution
- Scene: Railway pantograph (semi-static foreground + static background)

### Stage Timings

| Stage | Operation | Time | Throughput |
|-------|-----------|------|------------|
| 1 | Frame extraction (2 FPS) | 2 min | ~218 frames/min |
| 2.1 | Video chunking | 30 sec | 13,106 frames/min |
| 2.2 | Chunk extraction | 5 min | 1,311 frames/min |
| 2.3 | VDA depth (5 GPUs) | 35 min | 187 frames/min |
| 2.4 | Depth merging | 10 min | 655 frames/min |
| 3 | YOLO11 training | 150 min | - |
| 4 | Contact tracking | 15 min | 437 frames/min |
| 5 | 3D estimation | 5 min | 1,310 frames/min |
| 6 | Kalman filtering | 2 min | 3,276 frames/min |
| 7 | COLMAP | 30 min | 22 frames/min |
| 8 | Training (30k iter) | 120 min | - |
| 9 | USD export | 3 min | 2,184 frames/min |
| **Total** | **End-to-end** | **~6.5 hours** | **~ 17 frames/min** |

### Bottlenecks
1. **VDA depth estimation** (35 min, 46% of pipeline)
2. **Gaussian Splatting training** (120 min, but parallelizable with tracking)
3. **YOLO11 training** (150 min, one-time per scene type)

### Optimization Opportunities
- Run Gaussian Splatting training while tracking (saves 120 min)
- Cache YOLO11 model across similar scenes (saves 150 min per scene)
- Reduce VDA overlap to 16 frames (saves ~5 min)
- Use FP16 on GPUs with Tensor Cores (potential 2× speedup)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input: Raw Video                          │
│                    (train-mounted camera)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │  process_video  │
                   │  (FFmpeg)       │
                   └────────┬────────┘
                            │
                ┌───────────┴────────────┐
                │                        │
                ▼                        ▼
         ┌──────────┐            ┌──────────┐
         │  Frames  │            │ Metadata │
         │  (JPEG)  │            │  (JSON)  │
         └────┬─────┘            └──────────┘
              │
              ├──────────────┬──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │   VDA    │   │  YOLO11  │   │ COLMAP   │
       │  Depth   │   │Detection │   │  SfM     │
       └────┬─────┘   └────┬─────┘   └────┬─────┘
            │              │              │
            ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │  Depth   │   │2D Tracks │   │  Camera  │
       │   Maps   │   │(Bboxes)  │   │  Poses   │
       └────┬─────┘   └────┬─────┘   └────┬─────┘
            │              │              │
            └──────┬───────┴──────┬───────┘
                   │              │
                   ▼              ▼
            ┌──────────┐   ┌──────────┐
            │3D Tracks │   │Gaussian  │
            │(Fused)   │   │  Splat   │
            └────┬─────┘   └────┬─────┘
                 │              │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────┐
                 │   USD    │
                 │ Export   │
                 └────┬─────┘
                      │
                      ▼
              ┌───────────────┐
              │   USDView     │
              │ Visualization │
              └───────────────┘
```

---

## File Format Reference

### Depth Maps (PNG16)
- Format: 16-bit grayscale PNG
- Encoding: `uint16`
- Value range: [0, 65535]
- Depth representation: Inverse depth (1/z)
- Load: `cv2.imread(path, cv2.IMREAD_ANYDEPTH)`
- Normalize: `depth_norm = depth.astype(float) / 65535.0`

### Tracking JSON
```json
{
    "metadata": {
        "video_path": "string",
        "total_frames": "int",
        "fps": "float",
        "resolution": [width, height]
    },
    "tracks": [
        {
            "frame": "int",
            "bbox": [x1, y1, x2, y2],
            "confidence": "float",
            "contact_2d": [u, v],
            "vda_position_3d": {
                "x": "float",
                "y": "float",
                "z": "float",
                "depth_raw": "float",
                "depth_std": "float"
            }
        }
    ]
}
```

### Gaussian Splat PLY
```
ply
format binary_little_endian 1.0
element vertex <N>
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
<binary data>
```

---

## References

### Academic Papers
1. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
2. Yang et al., "Depth Anything V2" (CVPR 2025)
3. Jocher et al., "YOLO11: State-of-the-Art Object Detection" (2024)

### Software Dependencies
- FFmpeg: https://ffmpeg.org/
- COLMAP: https://colmap.github.io/
- Video Depth Anything: https://github.com/DepthAnything/Video-Depth-Anything
- Ultralytics YOLO11: https://github.com/ultralytics/ultralytics
- OpenUSD: https://openusd.org/

### Hardware References
- NVIDIA GTX 1660 Ti: https://www.nvidia.com/en-us/geforce/graphics-cards/gtx-1660-ti/
- Tensor Core whitepaper: https://www.nvidia.com/en-us/data-center/tensor-cores/

---

**Document Version:** 1.0  
**Last Updated:** November 17, 2025  
**Pipeline Version:** Based on 117 development sessions (Feb-Nov 2025)
