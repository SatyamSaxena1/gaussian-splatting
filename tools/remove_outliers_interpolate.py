#!/usr/bin/env python3
"""
Outlier Removal and Interpolation

Detects and removes outliers from tracking data, then interpolates
through gaps using cubic splines.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from scipy import interpolate
import matplotlib.pyplot as plt


def load_tracking_data(json_path):
    """Load tracking JSON data."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def detect_outliers_zscore(positions, threshold=3.0):
    """Detect outliers using z-score method."""
    # Z-scores for each dimension
    z_scores = np.abs((positions - np.mean(positions, axis=0)) / 
                      np.std(positions, axis=0))
    
    # Outliers if any dimension exceeds threshold
    outlier_mask = np.any(z_scores > threshold, axis=1)
    
    return outlier_mask, z_scores


def detect_outliers_velocity(positions, threshold=5.0):
    """Detect outliers using velocity magnitude."""
    velocities = np.diff(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Outliers if speed exceeds median + threshold * MAD
    median_speed = np.median(speeds)
    mad = np.median(np.abs(speeds - median_speed))
    outlier_speeds = speeds > median_speed + threshold * mad
    
    # Convert to position indices (add False at end)
    outlier_mask = np.append(outlier_speeds, False)
    
    return outlier_mask, speeds


def detect_outliers_acceleration(positions, threshold=5.0):
    """Detect outliers using acceleration magnitude."""
    velocities = np.diff(positions, axis=0)
    accelerations = np.diff(velocities, axis=0)
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    # Outliers if acceleration exceeds threshold
    median_accel = np.median(accel_magnitudes)
    mad = np.median(np.abs(accel_magnitudes - median_accel))
    outlier_accels = accel_magnitudes > median_accel + threshold * mad
    
    # Convert to position indices (add False at beginning and end)
    outlier_mask = np.append(np.append(False, outlier_accels), False)
    
    return outlier_mask, accel_magnitudes


def detect_outliers_combined(positions, z_threshold=3.0, vel_threshold=5.0, 
                             accel_threshold=5.0):
    """Combine multiple outlier detection methods."""
    outlier_zscore, _ = detect_outliers_zscore(positions, z_threshold)
    outlier_velocity, _ = detect_outliers_velocity(positions, vel_threshold)
    outlier_accel, _ = detect_outliers_acceleration(positions, accel_threshold)
    
    # Combine: outlier if flagged by any method
    combined_outliers = outlier_zscore | outlier_velocity | outlier_accel
    
    return combined_outliers


def interpolate_positions(frames, positions, valid_mask, method='cubic'):
    """
    Interpolate positions through gaps.
    
    Args:
        frames: Frame indices
        positions: Position array (may have NaN for invalid)
        valid_mask: Boolean mask of valid positions
        method: Interpolation method ('linear', 'cubic', 'quintic')
    
    Returns:
        interpolated_positions: Positions with gaps filled
    """
    if np.sum(valid_mask) < 4:
        print("Warning: Too few valid points for interpolation")
        return positions
    
    valid_frames = frames[valid_mask]
    valid_positions = positions[valid_mask]
    
    # Interpolate each dimension
    interpolated = np.zeros_like(positions)
    
    for dim in range(3):
        # Create interpolation function
        if method == 'cubic' and len(valid_frames) >= 4:
            f = interpolate.interp1d(valid_frames, valid_positions[:, dim],
                                    kind='cubic', bounds_error=False,
                                    fill_value='extrapolate')
        elif method == 'quintic' and len(valid_frames) >= 6:
            f = interpolate.interp1d(valid_frames, valid_positions[:, dim],
                                    kind='quintic', bounds_error=False,
                                    fill_value='extrapolate')
        else:
            # Fall back to linear
            f = interpolate.interp1d(valid_frames, valid_positions[:, dim],
                                    kind='linear', bounds_error=False,
                                    fill_value='extrapolate')
        
        # Interpolate all frames
        interpolated[:, dim] = f(frames)
    
    return interpolated


def smooth_with_moving_average(positions, window_size=5):
    """Apply moving average smoothing."""
    if window_size < 3:
        return positions
    
    smoothed = np.zeros_like(positions)
    half_window = window_size // 2
    
    for i in range(len(positions)):
        start = max(0, i - half_window)
        end = min(len(positions), i + half_window + 1)
        smoothed[i] = np.mean(positions[start:end], axis=0)
    
    return smoothed


def process_tracking_data(tracks, z_threshold=3.0, vel_threshold=5.0,
                          accel_threshold=5.0, interpolation_method='cubic',
                          apply_smoothing=False, smoothing_window=5):
    """
    Process tracking data: detect outliers, remove, interpolate.
    
    Returns:
        processed_tracks: Cleaned tracks
        outlier_info: Information about removed outliers
    """
    # Extract positions and frame indices
    positions = []
    frames = []
    detected_mask = []
    
    for i, track in enumerate(tracks):
        frames.append(i)
        if track['detected']:
            positions.append(track['contact_3d'])
            detected_mask.append(True)
        else:
            positions.append([np.nan, np.nan, np.nan])
            detected_mask.append(False)
    
    frames = np.array(frames)
    positions = np.array(positions)
    detected_mask = np.array(detected_mask)
    
    # Get valid positions (originally detected)
    valid_positions = positions[detected_mask]
    valid_frames = frames[detected_mask]
    
    if len(valid_positions) < 4:
        print("Warning: Too few valid detections for outlier removal")
        return tracks, {'outlier_count': 0}
    
    # Detect outliers in valid positions
    print("Detecting outliers...")
    outlier_mask = detect_outliers_combined(valid_positions, z_threshold,
                                           vel_threshold, accel_threshold)
    
    outlier_count = np.sum(outlier_mask)
    outlier_indices = valid_frames[outlier_mask]
    
    print(f"Found {outlier_count} outliers ({outlier_count/len(valid_positions)*100:.1f}%)")
    
    # Create final valid mask (detected and not outlier)
    final_valid_mask = detected_mask.copy()
    final_valid_mask[valid_frames[outlier_mask]] = False
    
    # Interpolate through all invalid positions (missing + outliers)
    print("Interpolating through gaps...")
    interpolated_positions = interpolate_positions(frames, positions,
                                                  final_valid_mask,
                                                  interpolation_method)
    
    # Optional smoothing
    if apply_smoothing:
        print(f"Applying moving average smoothing (window={smoothing_window})...")
        interpolated_positions = smooth_with_moving_average(interpolated_positions,
                                                           smoothing_window)
    
    # Create processed tracks
    processed_tracks = []
    
    for i, track in enumerate(tracks):
        track_copy = track.copy()
        
        # Update position
        track_copy['contact_3d'] = interpolated_positions[i].tolist()
        
        # Mark processing status
        if i in outlier_indices:
            track_copy['outlier_removed'] = True
            track_copy['contact_3d_original'] = track['contact_3d']
        
        if not detected_mask[i] or i in outlier_indices:
            track_copy['interpolated'] = True
            track_copy['detected'] = False  # Mark as not originally detected
        
        track_copy['processed'] = True
        
        # Recompute velocity
        if i > 0:
            vel = interpolated_positions[i] - interpolated_positions[i-1]
            track_copy['velocity_3d'] = vel.tolist()
        
        processed_tracks.append(track_copy)
    
    # Collect outlier info
    outlier_info = {
        'outlier_count': int(outlier_count),
        'outlier_indices': [int(x) for x in outlier_indices.tolist()],
        'outlier_percentage': float(outlier_count / len(valid_positions) * 100),
        'interpolation_method': interpolation_method,
        'smoothing_applied': apply_smoothing,
        'smoothing_window': int(smoothing_window) if apply_smoothing else None,
        'z_threshold': float(z_threshold),
        'velocity_threshold': float(vel_threshold),
        'acceleration_threshold': float(accel_threshold)
    }
    
    return processed_tracks, outlier_info


def compute_improvement_metrics(original_tracks, processed_tracks):
    """Compute metrics showing improvement after processing."""
    # Extract positions
    original_positions = []
    processed_positions = []
    
    for orig, proc in zip(original_tracks, processed_tracks):
        if orig['detected']:
            original_positions.append(orig['contact_3d'])
            processed_positions.append(proc['contact_3d'])
    
    original_positions = np.array(original_positions)
    processed_positions = np.array(processed_positions)
    
    # Compute metrics
    def compute_path_metrics(positions):
        segments = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        
        total_path = np.sum(segment_lengths)
        
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        accelerations = np.diff(velocities, axis=0)
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        return {
            'path_length': total_path,
            'mean_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'std_speed': np.std(speeds),
            'mean_accel': np.mean(accel_magnitudes),
            'max_accel': np.max(accel_magnitudes)
        }
    
    original_metrics = compute_path_metrics(original_positions)
    processed_metrics = compute_path_metrics(processed_positions)
    
    improvements = {
        'path_reduction': float((original_metrics['path_length'] - 
                          processed_metrics['path_length']) / 
                          original_metrics['path_length'] * 100),
        'max_speed_reduction': float((original_metrics['max_speed'] - 
                               processed_metrics['max_speed']) / 
                               original_metrics['max_speed'] * 100),
        'max_accel_reduction': float((original_metrics['max_accel'] - 
                               processed_metrics['max_accel']) / 
                               original_metrics['max_accel'] * 100),
        'speed_std_reduction': float((original_metrics['std_speed'] - 
                               processed_metrics['std_speed']) / 
                               original_metrics['std_speed'] * 100)
    }
    
    return original_metrics, processed_metrics, improvements


def plot_outlier_removal(original_tracks, processed_tracks, outlier_info, output_path):
    """Plot outlier removal results."""
    # Extract positions
    original_positions = []
    processed_positions = []
    frames = []
    outlier_flags = []
    
    for i, (orig, proc) in enumerate(zip(original_tracks, processed_tracks)):
        if orig['detected']:
            original_positions.append(orig['contact_3d'])
            processed_positions.append(proc['contact_3d'])
            frames.append(i)
            outlier_flags.append(proc.get('outlier_removed', False))
    
    original_positions = np.array(original_positions)
    processed_positions = np.array(processed_positions)
    frames = np.array(frames)
    outlier_flags = np.array(outlier_flags)
    time_s = frames / 30.0
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Position comparisons
    for i, (ax, label) in enumerate(zip(axes[:, 0], ['X', 'Y', 'Z'])):
        # Plot all points
        ax.plot(time_s, original_positions[:, i], 'r-', alpha=0.5,
               linewidth=1, label='Original')
        ax.plot(time_s, processed_positions[:, i], 'b-', alpha=0.8,
               linewidth=1.5, label='Processed')
        
        # Highlight outliers
        if np.any(outlier_flags):
            outlier_times = time_s[outlier_flags]
            outlier_values = original_positions[outlier_flags, i]
            ax.scatter(outlier_times, outlier_values, c='red', s=50,
                      marker='x', linewidths=2, label='Outliers', zorder=10)
        
        ax.set_ylabel(f'{label} Position (m)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{label}-Position: Outlier Removal')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Speed comparison
    original_vels = np.diff(original_positions, axis=0)
    processed_vels = np.diff(processed_positions, axis=0)
    original_speeds = np.linalg.norm(original_vels, axis=1)
    processed_speeds = np.linalg.norm(processed_vels, axis=1)
    
    axes[0, 1].plot(time_s[:-1], original_speeds, 'r-', alpha=0.5,
                   linewidth=1, label='Original')
    axes[0, 1].plot(time_s[:-1], processed_speeds, 'b-', alpha=0.8,
                   linewidth=1.5, label='Processed')
    axes[0, 1].set_ylabel('Speed (units/frame)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_title('Speed Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Acceleration comparison
    original_accels = np.diff(original_vels, axis=0)
    processed_accels = np.diff(processed_vels, axis=0)
    original_accel_mag = np.linalg.norm(original_accels, axis=1)
    processed_accel_mag = np.linalg.norm(processed_accels, axis=1)
    
    axes[1, 1].plot(time_s[:-2], original_accel_mag, 'r-', alpha=0.5,
                   linewidth=1, label='Original')
    axes[1, 1].plot(time_s[:-2], processed_accel_mag, 'b-', alpha=0.8,
                   linewidth=1.5, label='Processed')
    axes[1, 1].set_ylabel('Acceleration (units/frame²)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Acceleration Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[2, 1].axis('off')
    summary_text = f"""
OUTLIER REMOVAL SUMMARY

Outliers detected: {outlier_info['outlier_count']}
Percentage: {outlier_info['outlier_percentage']:.1f}%

Interpolation method: {outlier_info['interpolation_method']}
Smoothing applied: {outlier_info['smoothing_applied']}

Detection thresholds:
  Z-score: {outlier_info['z_threshold']:.1f}
  Velocity: {outlier_info['velocity_threshold']:.1f}
  Acceleration: {outlier_info['acceleration_threshold']:.1f}

Max speed:
  Before: {np.max(original_speeds):.2f}
  After: {np.max(processed_speeds):.2f}

Max acceleration:
  Before: {np.max(original_accel_mag):.2f}
  After: {np.max(processed_accel_mag):.2f}
"""
    axes[2, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Outlier removal plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Remove outliers and interpolate trajectory data')
    parser.add_argument('input_json', help='Input tracking JSON file')
    parser.add_argument('--output', '-o', help='Output processed JSON file')
    parser.add_argument('--z-threshold', type=float, default=3.0,
                       help='Z-score threshold for outlier detection (default: 3.0)')
    parser.add_argument('--velocity-threshold', type=float, default=5.0,
                       help='Velocity threshold for outlier detection (default: 5.0)')
    parser.add_argument('--accel-threshold', type=float, default=5.0,
                       help='Acceleration threshold for outlier detection (default: 5.0)')
    parser.add_argument('--interpolation', choices=['linear', 'cubic', 'quintic'],
                       default='cubic', help='Interpolation method (default: cubic)')
    parser.add_argument('--smooth', action='store_true',
                       help='Apply moving average smoothing after interpolation')
    parser.add_argument('--smooth-window', type=int, default=5,
                       help='Smoothing window size (default: 5)')
    parser.add_argument('--plot', help='Output plot path')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading tracking data from {args.input_json}...")
    data = load_tracking_data(args.input_json)
    
    print(f"Total frames: {len(data['tracks'])}")
    
    # Process data
    print("\nProcessing trajectory data...")
    processed_tracks, outlier_info = process_tracking_data(
        data['tracks'],
        z_threshold=args.z_threshold,
        vel_threshold=args.velocity_threshold,
        accel_threshold=args.accel_threshold,
        interpolation_method=args.interpolation,
        apply_smoothing=args.smooth,
        smoothing_window=args.smooth_window
    )
    
    # Compute improvements
    print("\nComputing improvement metrics...")
    original_metrics, processed_metrics, improvements = compute_improvement_metrics(
        data['tracks'], processed_tracks)
    
    print("\nImprovement Results:")
    print(f"  Path length reduction: {improvements['path_reduction']:.1f}%")
    print(f"  Max speed reduction: {improvements['max_speed_reduction']:.1f}%")
    print(f"  Max accel reduction: {improvements['max_accel_reduction']:.1f}%")
    print(f"  Speed std reduction: {improvements['speed_std_reduction']:.1f}%")
    
    # Save processed data
    output_path = args.output or args.input_json.replace('.json', '_cleaned.json')
    
    processed_data = data.copy()
    processed_data['tracks'] = processed_tracks
    processed_data['outlier_removal'] = outlier_info
    processed_data['improvement_metrics'] = improvements
    
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"\n✅ Processed tracking data saved to {output_path}")
    
    # Generate plot
    if args.plot:
        plot_path = args.plot
    else:
        plot_path = str(Path(output_path).parent / 'outlier_removal_comparison.png')
    
    plot_outlier_removal(data['tracks'], processed_tracks, outlier_info, plot_path)
    
    print("\n" + "="*70)
    print("OUTLIER REMOVAL COMPLETE")
    print("="*70)
    print(f"Outliers removed: {outlier_info['outlier_count']}")
    print(f"Path reduced by: {improvements['path_reduction']:.1f}%")
    print(f"Max speed reduced by: {improvements['max_speed_reduction']:.1f}%")
    print(f"Output: {output_path}")
    print(f"Plot: {plot_path}")


if __name__ == '__main__':
    main()
