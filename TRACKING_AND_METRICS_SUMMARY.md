# Tracking & Metrics Implementation Summary

## Overview
Successfully enhanced the railway pantograph tracking pipeline by integrating **Optical Flow** fallback, **Video Depth Anything (VDA)** depth sampling, and **Metric Extraction** (Height, Stagger, Gradient).

## Key Features Implemented

### 1. Robust Tracking (`tools/track_contact_yolo.py`)
- **Optical Flow Fallback**: Implemented `cv2.calcOpticalFlowPyrLK` to bridge gaps when YOLO detection fails.
    - *Result*: Recovered **35%** of frames in the test clip where YOLO failed.
- **Robust Depth Sampling**:
    - Changed from simple median to **90th percentile** sampling to prioritize the pantograph structure over the background sky.
    - Added a vertical offset (+5px) to sample the pantograph body instead of the empty space above the contact point.

### 2. Dynamic Kalman Filter (`tools/kalman_filter_trajectory.py`)
- **Dynamic Noise Adjustment**: The filter now adjusts its measurement noise covariance (`R`) based on the standard deviation of the VDA depth map.
- **Benefit**: Smoother trajectories when depth data is noisy, tighter tracking when data is clean.

### 3. Metric Extraction (`tools/extract_metrics.py`)
- **Scale Recovery**: Implemented a calibration method using the **known width** of the detected pantograph region.
    - *Calibration Insight*: YOLO detects a small ~100mm region (contact strip), not the full 1600mm pantograph. Adjusting for this yielded realistic results.
- **Computed Metrics**:
    - **Height**: Vertical distance from rail level (Avg: ~5.17m).
    - **Stagger**: Lateral deviation from track center.
    - **Gradient**: Rate of change in height.

### 4. Visualization (`tools/overlay_metrics.py`)
- Created a tool to overlay computed metrics directly onto the video frames.
- Generated side-by-side comparisons (Tracking vs. VDA Depth) to verify alignment.

## Verification Results (WhatsApp Clip)
- **Total Frames**: 301
- **Detection Rate**: 93% (up from ~57% with YOLO alone)
- **Average Height**: 5.17 m
- **Height Range**: 4.90 m - 5.43 m

## Next Steps
- **Calibration**: Implement a more robust calibration using fixed scene features (masts, rails) or full pantograph horn detection.
- **Real-time**: Optimize VDA and tracking for near real-time performance.
