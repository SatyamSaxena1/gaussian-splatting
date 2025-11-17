#!/usr/bin/env python3
"""
Comprehensive 3D Trajectory Analysis for Pantograph Contact Tracking

Analyzes contact point trajectories, velocities, and dynamics from YOLO tracking data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import signal, interpolate
from scipy.spatial.distance import euclidean
import sys


def load_tracking_data(json_path):
    """Load tracking JSON data."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def extract_trajectories(data):
    """Extract 3D trajectories from tracking data."""
    trajectories = {
        'frame_ids': [],
        'positions': [],
        'velocities': [],
        'confidences': [],
        'detected': [],
        'bbox_widths': [],
        'bbox_heights': []
    }
    
    for track in data['tracks']:
        trajectories['frame_ids'].append(track['frame'])
        trajectories['detected'].append(track['detected'])
        trajectories['confidences'].append(track.get('confidence', 0.0))
        
        if track['detected']:
            trajectories['positions'].append(track['contact_3d'])
            trajectories['velocities'].append(track['velocity_3d'])
            
            # Calculate bbox dimensions
            bbox = track['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            trajectories['bbox_widths'].append(width)
            trajectories['bbox_heights'].append(height)
        else:
            trajectories['positions'].append([np.nan, np.nan, np.nan])
            trajectories['velocities'].append([0.0, 0.0, 0.0])
            trajectories['bbox_widths'].append(np.nan)
            trajectories['bbox_heights'].append(np.nan)
    
    # Convert to numpy arrays
    trajectories['positions'] = np.array(trajectories['positions'])
    trajectories['velocities'] = np.array(trajectories['velocities'])
    trajectories['confidences'] = np.array(trajectories['confidences'])
    trajectories['detected'] = np.array(trajectories['detected'])
    trajectories['bbox_widths'] = np.array(trajectories['bbox_widths'])
    trajectories['bbox_heights'] = np.array(trajectories['bbox_heights'])
    
    return trajectories


def compute_motion_statistics(trajectories):
    """Compute detailed motion statistics."""
    positions = trajectories['positions']
    velocities = trajectories['velocities']
    
    # Filter valid detections
    valid_mask = trajectories['detected']
    valid_positions = positions[valid_mask]
    valid_velocities = velocities[valid_mask]
    
    # Speed (magnitude of velocity)
    speeds = np.linalg.norm(valid_velocities, axis=1)
    
    # Acceleration (derivative of velocity)
    accelerations = np.diff(valid_velocities, axis=0)
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    # Path length
    path_segments = np.diff(valid_positions, axis=0)
    segment_lengths = np.linalg.norm(path_segments, axis=1)
    total_path_length = np.sum(segment_lengths)
    
    # Displacement (start to end)
    displacement = euclidean(valid_positions[0], valid_positions[-1])
    
    # Tortuosity (path length / displacement)
    tortuosity = total_path_length / displacement if displacement > 0 else np.inf
    
    stats = {
        'speed': {
            'mean': np.mean(speeds),
            'std': np.std(speeds),
            'min': np.min(speeds),
            'max': np.max(speeds),
            'median': np.median(speeds),
            'p95': np.percentile(speeds, 95),
            'p99': np.percentile(speeds, 99)
        },
        'acceleration': {
            'mean': np.mean(accel_magnitudes),
            'std': np.std(accel_magnitudes),
            'max': np.max(accel_magnitudes)
        },
        'position_range': {
            'x': (np.min(valid_positions[:, 0]), np.max(valid_positions[:, 0])),
            'y': (np.min(valid_positions[:, 1]), np.max(valid_positions[:, 1])),
            'z': (np.min(valid_positions[:, 2]), np.max(valid_positions[:, 2]))
        },
        'position_std': {
            'x': np.std(valid_positions[:, 0]),
            'y': np.std(valid_positions[:, 1]),
            'z': np.std(valid_positions[:, 2])
        },
        'path_length': total_path_length,
        'displacement': displacement,
        'tortuosity': tortuosity,
        'valid_frames': np.sum(valid_mask),
        'total_frames': len(valid_mask)
    }
    
    return stats, speeds, accel_magnitudes


def detect_outliers(trajectories, z_threshold=3.0):
    """Detect outliers in trajectory data using z-score method."""
    positions = trajectories['positions']
    valid_mask = trajectories['detected']
    valid_positions = positions[valid_mask]
    
    # Z-scores for each dimension
    z_scores = np.abs((valid_positions - np.mean(valid_positions, axis=0)) / 
                      np.std(valid_positions, axis=0))
    
    # Outliers if any dimension exceeds threshold
    outlier_mask = np.any(z_scores > z_threshold, axis=1)
    outlier_indices = np.where(valid_mask)[0][outlier_mask]
    
    return outlier_indices, z_scores[outlier_mask]


def interpolate_gaps(trajectories, method='linear'):
    """Interpolate through detection gaps."""
    positions = trajectories['positions'].copy()
    valid_mask = trajectories['detected']
    
    # Get valid indices
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        return positions
    
    # Interpolate each dimension
    interpolated = positions.copy()
    for dim in range(3):
        valid_data = positions[valid_mask, dim]
        
        # Create interpolation function
        f = interpolate.interp1d(valid_indices, valid_data, 
                                kind=method, bounds_error=False, 
                                fill_value='extrapolate')
        
        # Fill gaps
        all_indices = np.arange(len(positions))
        interpolated[:, dim] = f(all_indices)
    
    return interpolated


def smooth_trajectory(positions, window_length=11, polyorder=3):
    """Smooth trajectory using Savitzky-Golay filter."""
    if len(positions) < window_length:
        window_length = len(positions) if len(positions) % 2 == 1 else len(positions) - 1
    
    smoothed = np.zeros_like(positions)
    for dim in range(3):
        smoothed[:, dim] = signal.savgol_filter(positions[:, dim], 
                                                window_length, polyorder)
    return smoothed


def analyze_frequency_components(positions, fps=30.0):
    """Analyze frequency components of motion."""
    # FFT for each dimension
    freq_analysis = {}
    
    for dim, name in enumerate(['X', 'Y', 'Z']):
        signal_data = positions[:, dim]
        
        # Remove mean (DC component)
        signal_data = signal_data - np.mean(signal_data)
        
        # Compute FFT
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1/fps)
        
        # Only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Find dominant frequency
        dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
        dominant_freq = positive_freqs[dominant_idx]
        
        freq_analysis[name] = {
            'freqs': positive_freqs,
            'magnitude': magnitude,
            'dominant_freq': dominant_freq,
            'dominant_magnitude': magnitude[dominant_idx]
        }
    
    return freq_analysis


def plot_comprehensive_analysis(trajectories, stats, speeds, output_dir):
    """Create comprehensive visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    positions = trajectories['positions']
    velocities = trajectories['velocities']
    confidences = trajectories['confidences']
    valid_mask = trajectories['detected']
    
    valid_positions = positions[valid_mask]
    valid_velocities = velocities[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # 1. 3D Trajectory Plot
    fig = plt.figure(figsize=(15, 5))
    
    # XY plane
    ax1 = fig.add_subplot(131)
    ax1.plot(valid_positions[:, 0], valid_positions[:, 1], 'b-', alpha=0.6, linewidth=1)
    ax1.scatter(valid_positions[::100, 0], valid_positions[::100, 1], 
               c=valid_indices[::100], cmap='viridis', s=20, alpha=0.8)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Contact Point Trajectory (XY Plane)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # XZ plane
    ax2 = fig.add_subplot(132)
    ax2.plot(valid_positions[:, 0], valid_positions[:, 2], 'r-', alpha=0.6, linewidth=1)
    ax2.scatter(valid_positions[::100, 0], valid_positions[::100, 2], 
               c=valid_indices[::100], cmap='viridis', s=20, alpha=0.8)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Contact Point Trajectory (XZ Plane)')
    ax2.grid(True, alpha=0.3)
    
    # YZ plane
    ax3 = fig.add_subplot(133)
    ax3.plot(valid_positions[:, 1], valid_positions[:, 2], 'g-', alpha=0.6, linewidth=1)
    ax3.scatter(valid_positions[::100, 1], valid_positions[::100, 2], 
               c=valid_indices[::100], cmap='viridis', s=20, alpha=0.8)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Contact Point Trajectory (YZ Plane)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Position over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    time_s = valid_indices / 30.0  # Assuming 30 fps
    
    for i, (ax, label, color) in enumerate(zip(axes, ['X', 'Y', 'Z'], ['b', 'g', 'r'])):
        ax.plot(time_s, valid_positions[:, i], color=color, linewidth=1, alpha=0.7)
        ax.set_ylabel(f'{label} Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Contact Point {label}-Position over Time')
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(output_dir / 'position_vs_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Velocity analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Speed over time
    axes[0, 0].plot(time_s, speeds, 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].axhline(stats['speed']['mean'], color='r', linestyle='--', 
                       label=f'Mean: {stats["speed"]["mean"]:.3f}')
    axes[0, 0].fill_between(time_s, 
                            stats['speed']['mean'] - stats['speed']['std'],
                            stats['speed']['mean'] + stats['speed']['std'],
                            alpha=0.3, color='r', label='±1 std')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed (units/frame)')
    axes[0, 0].set_title('Contact Point Speed over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Speed distribution
    axes[0, 1].hist(speeds, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(stats['speed']['mean'], color='r', linestyle='--', 
                       label=f'Mean: {stats["speed"]["mean"]:.3f}')
    axes[0, 1].axvline(stats['speed']['median'], color='g', linestyle='--', 
                       label=f'Median: {stats["speed"]["median"]:.3f}')
    axes[0, 1].set_xlabel('Speed (units/frame)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Speed Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity components
    axes[1, 0].plot(time_s, valid_velocities[:, 0], 'b-', alpha=0.7, label='Vx')
    axes[1, 0].plot(time_s, valid_velocities[:, 1], 'g-', alpha=0.7, label='Vy')
    axes[1, 0].plot(time_s, valid_velocities[:, 2], 'r-', alpha=0.7, label='Vz')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity (units/frame)')
    axes[1, 0].set_title('Velocity Components over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confidence over time
    axes[1, 1].plot(np.arange(len(confidences)) / 30.0, confidences, 
                    'purple', linewidth=1, alpha=0.7)
    axes[1, 1].axhline(np.mean(confidences[valid_mask]), color='r', linestyle='--',
                       label=f'Mean: {np.mean(confidences[valid_mask]):.3f}')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Detection Confidence')
    axes[1, 1].set_title('Detection Confidence over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'velocity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Detection gaps and quality
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Detection status
    detection_binary = valid_mask.astype(int)
    axes[0].fill_between(np.arange(len(detection_binary)) / 30.0, 0, detection_binary,
                         alpha=0.5, color='green', label='Detected')
    axes[0].fill_between(np.arange(len(detection_binary)) / 30.0, 
                         detection_binary, 1,
                         alpha=0.5, color='red', label='Missing')
    axes[0].set_ylabel('Detection Status')
    axes[0].set_title('Frame-by-Frame Detection Status')
    axes[0].legend()
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3)
    
    # Bounding box size over time
    valid_widths = trajectories['bbox_widths'][valid_mask]
    valid_heights = trajectories['bbox_heights'][valid_mask]
    axes[1].plot(time_s, valid_widths, 'b-', alpha=0.7, label='Width')
    axes[1].plot(time_s, valid_heights, 'r-', alpha=0.7, label='Height')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Bbox Size (pixels)')
    axes[1].set_title('Bounding Box Dimensions over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to {output_dir}/")


def generate_report(data, stats, outliers, output_path):
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append("3D TRAJECTORY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Dataset info
    report.append("DATASET INFORMATION:")
    report.append(f"  Total Frames: {stats['total_frames']}")
    report.append(f"  Valid Detections: {stats['valid_frames']} ({stats['valid_frames']/stats['total_frames']*100:.1f}%)")
    report.append(f"  Missing Frames: {stats['total_frames'] - stats['valid_frames']}")
    report.append(f"  Model: {data['metadata']['model']}")
    report.append(f"  Average Confidence: {data['metadata']['avg_confidence']:.3f}")
    report.append("")
    
    # Position statistics
    report.append("POSITION STATISTICS:")
    for axis in ['x', 'y', 'z']:
        range_min, range_max = stats['position_range'][axis]
        std = stats['position_std'][axis]
        report.append(f"  {axis.upper()}-axis:")
        report.append(f"    Range: [{range_min:.3f}, {range_max:.3f}] meters")
        report.append(f"    Span: {range_max - range_min:.3f} meters")
        report.append(f"    Std Dev: {std:.3f} meters")
    report.append("")
    
    # Motion statistics
    report.append("MOTION STATISTICS:")
    report.append(f"  Speed (units/frame):")
    report.append(f"    Mean: {stats['speed']['mean']:.4f}")
    report.append(f"    Std: {stats['speed']['std']:.4f}")
    report.append(f"    Median: {stats['speed']['median']:.4f}")
    report.append(f"    Min: {stats['speed']['min']:.4f}")
    report.append(f"    Max: {stats['speed']['max']:.4f}")
    report.append(f"    95th percentile: {stats['speed']['p95']:.4f}")
    report.append(f"    99th percentile: {stats['speed']['p99']:.4f}")
    report.append("")
    
    report.append(f"  Acceleration (units/frame²):")
    report.append(f"    Mean: {stats['acceleration']['mean']:.4f}")
    report.append(f"    Std: {stats['acceleration']['std']:.4f}")
    report.append(f"    Max: {stats['acceleration']['max']:.4f}")
    report.append("")
    
    # Path characteristics
    report.append("PATH CHARACTERISTICS:")
    report.append(f"  Total Path Length: {stats['path_length']:.3f} meters")
    report.append(f"  Straight-line Displacement: {stats['displacement']:.3f} meters")
    report.append(f"  Tortuosity: {stats['tortuosity']:.3f}")
    report.append(f"    (1.0 = straight line, >1.0 = curved path)")
    report.append("")
    
    # Outlier detection
    report.append("OUTLIER DETECTION:")
    report.append(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        report.append(f"  Outlier frames: {outliers[:20].tolist()}")
        if len(outliers) > 20:
            report.append(f"  ... and {len(outliers) - 20} more")
    report.append("")
    
    # Data quality assessment
    report.append("DATA QUALITY ASSESSMENT:")
    detection_rate = stats['valid_frames'] / stats['total_frames']
    if detection_rate > 0.95:
        quality = "EXCELLENT"
    elif detection_rate > 0.90:
        quality = "GOOD"
    elif detection_rate > 0.80:
        quality = "FAIR"
    else:
        quality = "POOR"
    report.append(f"  Detection Rate: {detection_rate*100:.1f}% ({quality})")
    
    if stats['speed']['max'] > 10 * stats['speed']['mean']:
        report.append(f"  ⚠️  WARNING: Large speed spikes detected (max/mean ratio: {stats['speed']['max']/stats['speed']['mean']:.1f})")
    
    if stats['tortuosity'] > 2.0:
        report.append(f"  ⚠️  WARNING: High path tortuosity suggests noisy tracking")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print(report_text)
    
    return report_text


def generate_next_steps(stats, outliers, output_path):
    """Generate actionable next steps based on analysis."""
    steps = []
    steps.append("=" * 80)
    steps.append("RECOMMENDED NEXT STEPS")
    steps.append("=" * 80)
    steps.append("")
    
    detection_rate = stats['valid_frames'] / stats['total_frames']
    
    # Data quality improvements
    steps.append("1. DATA QUALITY IMPROVEMENTS:")
    if detection_rate < 0.95:
        steps.append("   • Train model for more epochs (100+) to improve recall")
        steps.append("   • Use temporal interpolation to fill detection gaps")
        steps.append("   • Consider multi-frame tracking (e.g., DeepSORT, ByteTrack)")
    else:
        steps.append("   ✓ Detection rate is excellent (>95%)")
    
    if len(outliers) > stats['valid_frames'] * 0.05:
        steps.append(f"   • Remove {len(outliers)} outlier frames using z-score filtering")
        steps.append("   • Apply median filtering to reduce noise")
    steps.append("")
    
    # Trajectory smoothing
    steps.append("2. TRAJECTORY SMOOTHING:")
    steps.append("   • Apply Kalman filter for optimal state estimation")
    steps.append("   • Use Savitzky-Golay filter for polynomial smoothing")
    steps.append("   • Implement spline interpolation for smooth curves")
    steps.append("   • Test different smoothing parameters and validate visually")
    steps.append("")
    
    # Physical validation
    steps.append("3. PHYSICAL VALIDATION:")
    steps.append("   • Manually annotate 20-50 frames for ground truth")
    steps.append("   • Calculate RMSE between predictions and ground truth")
    steps.append("   • Verify physical plausibility of speeds and accelerations")
    if stats['speed']['max'] > 10 * stats['speed']['mean']:
        steps.append("   ⚠️  CRITICAL: Validate high-speed frames (likely artifacts)")
    steps.append("")
    
    # Scale calibration
    steps.append("4. SCALE CALIBRATION:")
    steps.append("   • Monocular depth has arbitrary scale - needs calibration")
    steps.append("   • Use known pantograph dimensions for scale correction")
    steps.append("   • Consider stereo camera setup for accurate depth")
    steps.append("   • Validate Z-axis measurements (current range suspiciously large)")
    steps.append("")
    
    # Contact dynamics analysis
    steps.append("5. CONTACT DYNAMICS ANALYSIS:")
    steps.append("   • Identify contact/separation events from Z-position")
    steps.append("   • Measure contact duration and frequency")
    steps.append("   • Analyze vertical oscillation patterns")
    steps.append("   • Compare with catenary wire simulations")
    steps.append("")
    
    # Simulation comparison
    steps.append("6. SIMULATION VALIDATION:")
    steps.append("   • Export trajectories to simulation format")
    steps.append("   • Compare tracked vs. simulated contact positions")
    steps.append("   • Validate force estimates from acceleration data")
    steps.append("   • Identify discrepancies for model improvement")
    steps.append("")
    
    # Advanced analysis
    steps.append("7. ADVANCED ANALYSIS:")
    steps.append("   • Frequency analysis (FFT) for periodic motion detection")
    steps.append("   • Modal analysis for vibration modes")
    steps.append("   • Contact pressure estimation from dynamics")
    steps.append("   • Wear prediction from contact patterns")
    steps.append("")
    
    # Code implementation
    steps.append("8. IMPLEMENTATION PRIORITIES:")
    steps.append("   Priority 1 (Immediate):")
    steps.append("     - Kalman filtering for trajectory smoothing")
    steps.append("     - Outlier removal and gap interpolation")
    steps.append("     - Ground truth validation (20 frames)")
    steps.append("")
    steps.append("   Priority 2 (Short-term):")
    steps.append("     - Scale calibration using known dimensions")
    steps.append("     - Contact event detection algorithm")
    steps.append("     - Comparison with simulation data")
    steps.append("")
    steps.append("   Priority 3 (Long-term):")
    steps.append("     - Multi-camera 3D reconstruction")
    steps.append("     - Real-time tracking deployment")
    steps.append("     - Automated anomaly detection")
    steps.append("")
    
    steps.append("=" * 80)
    steps.append("")
    steps.append("RECOMMENDED TOOLS TO IMPLEMENT:")
    steps.append("  1. kalman_filter_trajectory.py - State estimation")
    steps.append("  2. interpolate_gaps.py - Fill missing detections")
    steps.append("  3. detect_contact_events.py - Contact/separation analysis")
    steps.append("  4. calibrate_scale.py - Depth scale correction")
    steps.append("  5. compare_with_simulation.py - Validation against model")
    steps.append("")
    steps.append("=" * 80)
    
    steps_text = "\n".join(steps)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(steps_text)
    
    print(steps_text)
    
    return steps_text


def main():
    parser = argparse.ArgumentParser(description='Analyze 3D pantograph tracking trajectories')
    parser.add_argument('tracking_json', help='Path to tracking JSON file')
    parser.add_argument('--output-dir', default='trajectory_analysis',
                       help='Output directory for plots and reports')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Video frame rate (default: 30.0)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading tracking data from {args.tracking_json}...")
    data = load_tracking_data(args.tracking_json)
    
    # Extract trajectories
    print("Extracting trajectory data...")
    trajectories = extract_trajectories(data)
    
    # Compute statistics
    print("Computing motion statistics...")
    stats, speeds, accel_magnitudes = compute_motion_statistics(trajectories)
    
    # Detect outliers
    print("Detecting outliers...")
    outlier_indices, outlier_z_scores = detect_outliers(trajectories)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Creating visualization plots...")
    plot_comprehensive_analysis(trajectories, stats, speeds, output_dir)
    
    # Generate reports
    print("Generating analysis report...")
    report_path = output_dir / 'trajectory_analysis_report.txt'
    generate_report(data, stats, outlier_indices, report_path)
    
    print("Generating next steps plan...")
    steps_path = output_dir / 'next_steps_plan.txt'
    generate_next_steps(stats, outlier_indices, steps_path)
    
    # Save processed data
    print("Saving processed data...")
    np.savez(output_dir / 'trajectory_data.npz',
             positions=trajectories['positions'],
             velocities=trajectories['velocities'],
             confidences=trajectories['confidences'],
             detected=trajectories['detected'],
             speeds=speeds,
             outlier_indices=outlier_indices)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"   - trajectory_analysis_report.txt")
    print(f"   - next_steps_plan.txt")
    print(f"   - 3d_trajectories.png")
    print(f"   - position_vs_time.png")
    print(f"   - velocity_analysis.png")
    print(f"   - detection_quality.png")
    print(f"   - trajectory_data.npz")


if __name__ == '__main__':
    main()
