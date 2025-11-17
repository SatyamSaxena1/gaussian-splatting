# 🎥 Video Processing Guide for Gaussian Splatting

## Your Video Information

- **Location**: `/home/akash_gemperts/Downloads/output_webcam_20250205_151016.mp4`
- **Resolution**: 1920x1080
- **Duration**: ~20 minutes (1192 seconds)
- **Frame Rate**: 25 fps

## 🚀 Quick Start (Recommended)

Run the automated processing script:

```bash
cd /home/akash_gemperts/gaussian-splatting
./process_webcam_video.sh
```

This will:
1. Extract frames from your video (at 2 fps = ~2400 frames)
2. Scale them to 960x540 for faster processing
3. Save them to `data/webcam_scene/input/`

## 📋 Complete Workflow

### Step 1: Extract Frames from Video

You have several options for frame extraction:

#### Option A: Quick Processing (Recommended)
```bash
./process_webcam_video.sh
```

#### Option B: Custom Settings
```bash
./process_video.sh ~/Downloads/output_webcam_20250205_151016.mp4 ./data/my_scene 2 2
```

Parameters:
- `2` (3rd argument) = Extract 2 frames per second
- `2` (4th argument) = Scale to 1/2 resolution (960x540)

#### Option C: Manual Extraction
```bash
# Create directory
mkdir -p data/webcam_scene/input

# Extract frames (adjust fps and scale as needed)
ffmpeg -i ~/Downloads/output_webcam_20250205_151016.mp4 \
       -vf "fps=2,scale=960:540" \
       -qscale:v 2 \
       data/webcam_scene/input/frame_%04d.jpg
```

### Step 2: Run COLMAP for Camera Pose Estimation

Install COLMAP if needed:
```bash
sudo apt-get update
sudo apt-get install colmap
```

Process the frames with COLMAP:
```bash
cd /home/akash_gemperts/gaussian-splatting
python convert.py -s data/webcam_scene
```

This will:
- Extract image features
- Match features between frames
- Estimate camera poses
- Create sparse 3D reconstruction
- Undistort images

⏱️ **Time estimate**: 30-60 minutes depending on frame count

### Step 3: Train Gaussian Splatting Model

First, ensure the environment is set up:
```bash
# If not already done, run setup
./quick_setup.sh

# Activate environment
conda activate gaussian_splatting
```

Start training:
```bash
# Basic training
python train.py -s data/webcam_scene

# With memory optimization (recommended if VRAM < 24GB)
python train.py -s data/webcam_scene --data_device cpu

# With faster training (if you set up the accelerated rasterizer)
python train.py -s data/webcam_scene --optimizer_type sparse_adam --data_device cpu
```

⏱️ **Time estimate**: 1-4 hours depending on GPU and settings

### Step 4: View Results

After training completes:

```bash
# Render the trained model
python render.py -m output/<your_model_directory>

# View with the real-time viewer (if built)
./SIBR_viewers/bin/SIBR_gaussianViewer_app -m output/<your_model_directory>
```

## ⚙️ Recommended Settings

### For Best Quality
- Extract 3-5 fps
- Use original resolution (scale=1)
- Train for 30,000 iterations (default)

```bash
./process_video.sh ~/Downloads/output_webcam_20250205_151016.mp4 ./data/webcam_hq 3 1
```

### For Fast Testing
- Extract 1-2 fps
- Use 1/4 resolution (scale=4)
- Train for 7,000 iterations

```bash
./process_video.sh ~/Downloads/output_webcam_20250205_151016.mp4 ./data/webcam_test 1 4
python train.py -s data/webcam_test --iterations 7000 --data_device cpu
```

### For Balanced Quality/Speed (Recommended)
- Extract 2 fps (~2400 frames)
- Use 1/2 resolution (960x540)
- Use default training settings

```bash
./process_webcam_video.sh
python train.py -s data/webcam_scene --data_device cpu
```

## 💡 Important Tips for Video Processing

### 1. **Camera Movement**
For best results, the video should have:
- Smooth camera movements
- Good coverage of the scene from multiple angles
- Minimal motion blur
- Consistent lighting

### 2. **Frame Selection**
- Too few frames: Poor reconstruction
- Too many frames: Slower processing, diminishing returns
- Sweet spot: 1-3 fps for videos with smooth motion

### 3. **Memory Management**
If you run out of memory:
```bash
# Use CPU for image data
python train.py -s data/webcam_scene --data_device cpu

# Reduce density
python train.py -s data/webcam_scene --densify_grad_threshold 0.0005

# Skip test iterations
python train.py -s data/webcam_scene --test_iterations -1
```

### 4. **COLMAP Failures**
If COLMAP fails to reconstruct:
- Try different frame rates
- Ensure frames have good texture/features
- Check that camera moves enough between frames
- Reduce the number of frames if too many

## 📊 Expected Timeline

For your 20-minute video:

| Stage | Time | Output |
|-------|------|--------|
| Frame extraction (2fps) | 5-10 min | ~2400 frames |
| COLMAP processing | 30-60 min | Camera poses + sparse point cloud |
| Training | 1-4 hours | Trained Gaussian Splatting model |
| **Total** | **2-5 hours** | Ready to render! |

## 🔍 Troubleshooting

### Problem: "Too many frames, COLMAP is too slow"
**Solution**: Reduce fps or use frame selection
```bash
# Extract fewer frames
./process_video.sh <video> <output> 1 2  # 1 fps instead of 2
```

### Problem: "Out of GPU memory during training"
**Solution**: Use memory optimization flags
```bash
python train.py -s data/webcam_scene --data_device cpu --test_iterations -1
```

### Problem: "COLMAP reconstruction failed"
**Solution**: 
1. Check if frames have sufficient overlap
2. Try extracting frames at different intervals
3. Ensure good lighting and texture in video

### Problem: "Training is too slow"
**Solution**: Use training acceleration
```bash
# First, install accelerated rasterizer (see SETUP_INSTRUCTIONS.md)
pip uninstall diff-gaussian-rasterization -y
cd submodules/diff-gaussian-rasterization
git checkout 3dgs_accel
pip install .

# Then train with sparse adam
python train.py -s data/webcam_scene --optimizer_type sparse_adam
```

## 📁 Output Structure

After processing, your directory will look like:

```
data/webcam_scene/
├── input/                    # Extracted frames
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── distorted/               # COLMAP working directory
│   ├── database.db
│   └── sparse/
├── sparse/                  # Final camera poses
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── images/                  # Undistorted images (used for training)
    ├── frame_0001.jpg
    └── ...
```

## 🎯 Next Steps After Training

1. **Render views**: Generate novel views from your model
2. **Export**: Export to other formats for web viewing
3. **Optimize**: Further compress the model if needed
4. **Share**: Use web viewers to share your 3D scene

For more details, see the main README.md
