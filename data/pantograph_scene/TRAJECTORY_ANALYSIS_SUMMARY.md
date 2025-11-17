# 3D Trajectory Analysis Summary
## Pantograph Contact Point Tracking

**Date:** November 13, 2025  
**Dataset:** WhatsApp train video (6,552 frames @ 30 fps)  
**Model:** YOLOv11s trained for 50 epochs

---

## Executive Summary

Comprehensive analysis of 3D contact point trajectories reveals **excellent detection quality (99.0%)** but identifies **critical issues with depth scale and noise** that require immediate attention before simulation validation.

### Key Findings

✅ **Strengths:**
- 99.0% detection rate (6,484/6,552 frames)
- 90.8% average confidence
- Excellent detection consistency

⚠️ **Critical Issues:**
1. **Depth scale problem:** Z-axis spans 153 meters (unrealistic for pantograph)
2. **High noise:** 252 outlier frames (3.9%) with extreme speed spikes
3. **Extreme tortuosity:** Path length 4,719m vs displacement 4.3m (ratio: 1,109x)

---

## Detailed Analysis Results

### 1. Position Statistics

| Axis | Range (m) | Span (m) | Std Dev (m) | Assessment |
|------|-----------|----------|-------------|------------|
| **X** | -2.267 to 0.495 | 2.76 | 0.169 | ✅ Reasonable |
| **Y** | -19.317 to -0.148 | 19.17 | 0.613 | ⚠️ Large (likely camera motion) |
| **Z** | 1.022 to 154.540 | **153.52** | 4.235 | ❌ **UNREALISTIC** |

**Interpretation:**
- X-axis (horizontal): ~2.8m span is plausible for lateral pantograph motion
- Y-axis (vertical in frame): 19m suggests uncalibrated depth + camera motion
- Z-axis (depth): 153m is physically impossible - indicates depth scale error

### 2. Motion Dynamics

**Speed Statistics (units/frame):**
```
Mean:     0.727
Median:   0.271
Std Dev:  3.093
Max:      112.69  ⚠️ SPIKE
95th %:   2.130
99th %:   7.813
```

**Key Observations:**
- Median speed (0.27) much lower than mean (0.73) → heavy-tailed distribution
- **Max speed 155× the mean** → outliers/artifacts dominate
- 95th percentile (2.13) is reasonable → most data is good

**Acceleration Statistics (units/frame²):**
```
Mean:     1.167
Std Dev:  5.231
Max:      222.63  ⚠️ EXTREME
```

### 3. Path Characteristics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Path Length** | 4,718.8 m | Sum of all frame-to-frame movements |
| **Displacement** | 4.3 m | Straight-line start-to-end distance |
| **Tortuosity** | 1,109 | Path length / displacement |

**Analysis:**
- Tortuosity >> 1 indicates **severe noise** in tracking
- Should be 1.0-2.0 for smooth motion along wire
- Current value suggests position jitter dominates real motion

### 4. Outlier Detection

- **252 outliers detected** (3.9% of valid frames)
- Z-score threshold: 3.0 standard deviations
- Frames with extreme position jumps relative to trajectory

**Sample outlier frames:** 12, 662, 665-668, 741, 743, 747-759, ...

---

## Root Cause Analysis

### Problem 1: Monocular Depth Ambiguity
**Cause:** MiDaS produces depth in arbitrary scale (normalized 0-1)  
**Effect:** Z-axis spans 153 meters instead of ~1-2 meters  
**Solution:** Scale calibration using known pantograph dimensions

### Problem 2: Detection Bbox Jitter
**Cause:** Frame-to-frame bbox variation → position instability  
**Effect:** High-frequency noise in trajectory  
**Solution:** Temporal smoothing (Kalman filter, moving average)

### Problem 3: Outlier Artifacts
**Cause:** Occasional detection failures or bbox jumps  
**Effect:** Extreme speed spikes (112.7 units/frame)  
**Solution:** Outlier removal + interpolation

---

## Next Steps Plan

### Priority 1: Immediate (< 1 day)

#### 1.1 Scale Calibration
**Tool:** `calibrate_depth_scale.py`
```python
# Use known pantograph height (~1.5m) to calibrate
# Z-axis should span 0.5-2.0m for vertical oscillation
```

**Expected improvement:**
- Z-axis: 1.022-154.54m → 0.5-2.0m (realistic range)
- Speeds scaled to physical units (m/s)

#### 1.2 Trajectory Smoothing
**Tool:** `kalman_filter_trajectory.py`
```python
# Kalman filter parameters:
# Process noise: low (pantograph has inertia)
# Measurement noise: medium (bbox jitter)
```

**Expected improvement:**
- Tortuosity: 1,109 → 1.5-3.0 (smooth curve)
- Speed max: 112.7 → 5-10 (remove spikes)
- Std dev reduced by 50-70%

#### 1.3 Outlier Removal
**Tool:** `remove_outliers_interpolate.py`
```python
# 1. Remove 252 outliers (z-score > 3)
# 2. Interpolate gaps using cubic spline
# 3. Re-smooth with moving average
```

**Expected improvement:**
- 252 outliers → 0 (interpolated)
- Max speed: 112.7 → <10 (realistic)

### Priority 2: Short-term (1-3 days)

#### 2.1 Ground Truth Validation
**Action:** Manual annotation of 20-50 frames
- Select frames across video (uniform sampling)
- Precisely mark contact point in each frame
- Compute RMSE: predicted vs ground truth

**Metrics to validate:**
- 2D position accuracy (pixels)
- 3D position accuracy (meters, after calibration)
- Detection consistency

#### 2.2 Contact Event Detection
**Tool:** `detect_contact_events.py`
```python
# Analyze Z-position oscillations:
# - Contact: Z minimum (wire pressed down)
# - Separation: Z maximum (wire rebounds)
# - Frequency: ~5-20 Hz typical
```

**Output:**
- Contact event timestamps
- Contact duration statistics
- Oscillation frequency spectrum

#### 2.3 Frequency Analysis
**Tool:** Already in `analyze_3d_trajectories.py` (FFT analysis)
```python
# Identify periodic motion components:
# - Low freq (0.1-1 Hz): Camera/train motion
# - Mid freq (1-10 Hz): Pantograph dynamics
# - High freq (>10 Hz): Noise/vibration
```

### Priority 3: Long-term (1-2 weeks)

#### 3.1 Simulation Comparison
**Tool:** `compare_with_simulation.py`
```python
# Export to simulation format:
# - Timestamps
# - 3D positions (calibrated)
# - Velocities (filtered)
# - Contact events
#
# Compare metrics:
# - Position RMSE
# - Velocity correlation
# - Contact timing agreement
```

#### 3.2 Multi-Camera 3D Reconstruction
**Action:** Consider upgrading to stereo/multi-view
- Current: Monocular (depth ambiguity)
- Upgrade: Stereo pair (metric depth)
- Benefits: No scale calibration needed, sub-cm accuracy

#### 3.3 Real-time Deployment
**Optimization:**
- Convert YOLOv11s to TensorRT
- Implement streaming inference
- Deploy Kalman filter in real-time
- Target: 30+ fps on single GPU

---

## Recommended Tools to Implement

### Tool 1: `calibrate_depth_scale.py`
**Purpose:** Scale monocular depth to metric units

**Input:**
- Tracking JSON with uncalibrated depth
- Known dimension (pantograph height or width)

**Output:**
- Calibration scale factor
- Rescaled trajectory JSON

**Algorithm:**
1. Measure bbox height in pixels
2. Assume constant physical height (1.5m typical)
3. Compute scale: `physical_height / (depth * bbox_height)`
4. Apply scale to all Z-coordinates

### Tool 2: `kalman_filter_trajectory.py`
**Purpose:** Optimal state estimation with noise filtering

**Parameters:**
```python
# State: [x, y, z, vx, vy, vz]
# Measurement: [x, y, z]
process_noise = 0.1      # Pantograph has inertia
measurement_noise = 1.0  # Bbox jitter
```

**Benefits:**
- Removes high-frequency noise
- Fills short gaps automatically
- Provides velocity estimates
- Confidence intervals

### Tool 3: `detect_contact_events.py`
**Purpose:** Identify contact/separation from Z-position

**Algorithm:**
1. Find local minima in Z (contact points)
2. Find local maxima in Z (separation points)
3. Measure time between events (contact duration)
4. Compute statistics (frequency, duration distributions)

**Output:**
```json
{
  "contact_events": [
    {"frame": 100, "time": 3.33, "z": 1.2, "type": "contact"},
    {"frame": 150, "time": 5.00, "z": 1.8, "type": "separation"}
  ],
  "statistics": {
    "contact_frequency": 8.5,  // Hz
    "avg_contact_duration": 0.12,  // seconds
    "avg_separation_duration": 0.06
  }
}
```

### Tool 4: `remove_outliers_interpolate.py`
**Purpose:** Clean trajectory by removing artifacts

**Method:**
1. Detect outliers (z-score > 3 or velocity > threshold)
2. Mark as invalid
3. Interpolate using cubic spline
4. Optional: Apply additional smoothing

### Tool 5: `compare_with_simulation.py`
**Purpose:** Validation against physics simulation

**Input:**
- Cleaned trajectory (JSON)
- Simulation output (matching format)

**Analysis:**
- Position RMSE
- Velocity correlation
- Contact timing agreement
- Force estimate validation (from acceleration)

---

## Visualization Outputs

Generated plots saved to `data/pantograph_scene/trajectory_analysis/`:

1. **3d_trajectories.png** (119 KB)
   - XY, XZ, YZ projections
   - Color-coded by time
   - Shows overall path shape

2. **position_vs_time.png** (188 KB)
   - X, Y, Z positions over time
   - Reveals noise patterns
   - Shows scale issues clearly

3. **velocity_analysis.png** (183 KB)
   - Speed distribution
   - Velocity components
   - Detection confidence
   - Speed spikes visible

4. **detection_quality.png** (115 KB)
   - Frame-by-frame detection status
   - Bounding box size variation
   - Identifies missing frames

---

## Data Products

### Current Files
```
data/pantograph_scene/
├── contact_track_yolo.json          3.0 MB  Raw tracking data
├── contact_visualizations/          728 MB  Annotated frames (6,552)
├── trajectory_analysis/
│   ├── 3d_trajectories.png          119 KB  Path visualizations
│   ├── position_vs_time.png         188 KB  Time series plots
│   ├── velocity_analysis.png        183 KB  Motion analysis
│   ├── detection_quality.png        115 KB  Quality metrics
│   ├── trajectory_data.npz          419 KB  Processed arrays
│   ├── trajectory_analysis_report.txt 1.6 KB Statistics
│   └── next_steps_plan.txt          2.7 KB  Action items
```

### Files to Create
```
data/pantograph_scene/
├── contact_track_calibrated.json    Depth-scaled trajectory
├── contact_track_filtered.json      Kalman-filtered trajectory
├── contact_events.json              Contact/separation events
├── ground_truth_samples.json        Manual annotations
└── simulation_comparison.json       Validation results
```

---

## Critical Warnings

### ⚠️ DEPTH SCALE ISSUE
**Current:** Z-axis spans 1-154 meters (physically impossible)  
**Expected:** Z-axis should span 0.5-2.0 meters for pantograph oscillation  
**Action:** MUST calibrate depth scale before simulation comparison

### ⚠️ NOISE DOMINATES SIGNAL
**Current:** Tortuosity = 1,109 (path 1000× longer than displacement)  
**Expected:** Tortuosity < 3 for smooth motion  
**Action:** MUST apply filtering before extracting dynamics

### ⚠️ OUTLIERS PRESENT
**Current:** 252 frames (3.9%) with extreme position jumps  
**Expected:** <0.5% outliers in good tracking  
**Action:** Remove outliers and interpolate before analysis

---

## Success Metrics

### After Priority 1 Tasks (Immediate)
- [ ] Z-axis range: 0.5-2.0 m (currently 1-154 m)
- [ ] Tortuosity: 1.5-3.0 (currently 1,109)
- [ ] Max speed: <10 units/frame (currently 112.7)
- [ ] Outliers: 0% (currently 3.9%)
- [ ] Position std dev reduced by >50%

### After Priority 2 Tasks (Short-term)
- [ ] Ground truth RMSE: <5 pixels (2D), <0.1m (3D)
- [ ] Contact events detected and validated
- [ ] Frequency spectrum analyzed and physical
- [ ] Ready for simulation comparison

### After Priority 3 Tasks (Long-term)
- [ ] Simulation RMSE: <0.2m position, <0.5m/s velocity
- [ ] Real-time tracking deployed (30+ fps)
- [ ] Multi-camera system operational (if feasible)

---

## Conclusion

The tracking system successfully detects pantograph components with 99% reliability, but the 3D trajectory data requires **scale calibration and noise filtering** before it can be used for simulation validation.

**Immediate priority:** Implement the three Priority 1 tools to transform the raw noisy data into clean, physically meaningful trajectories ready for engineering analysis.

**Expected timeline:**
- Priority 1 tools: 4-8 hours implementation
- Ground truth validation: 2-4 hours annotation + analysis
- Simulation comparison: 1-2 days (depends on simulation format)

**Ready to proceed with implementation?** Let me know which tool you'd like to build first!
