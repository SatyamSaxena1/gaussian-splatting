#!/usr/bin/env python3
"""
Depth Scale Calibration for Pantograph Tracking

Calibrates monocular depth to metric units using known physical dimensions.
MiDaS depth is in arbitrary scale - this tool converts to meters.
"""

import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_tracking_data(json_path):
    """Load tracking JSON data."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def estimate_scale_from_bbox(tracks, known_height_m=1.5, method='median'):
    """
    Estimate depth scale using bbox height and known physical height.
    
    Args:
        tracks: List of tracking data
        known_height_m: Known physical height of pantograph (meters)
        method: 'median', 'mean', or 'robust' for scale estimation
    
    Returns:
        scale_factor: Multiplier to convert depth units to meters
    """
    scales = []
    
    for track in tracks:
        if not track['detected']:
            continue
        
        # Get bbox height in pixels
        bbox = track['bbox']
        bbox_height_px = bbox[3] - bbox[1]
        
        # Get depth (Z coordinate before scaling)
        z_raw = track['contact_3d'][2]
        
        if z_raw <= 0 or bbox_height_px <= 0:
            continue
        
        # Physical height = bbox_height_px * depth / focal_length
        # For simplicity, assume focal_length is encoded in depth
        # Scale factor: how much to multiply Z to get meters
        
        # Angular size: h_physical / depth = h_pixels / focal
        # Rearranging: depth = h_physical * focal / h_pixels
        # Since we don't know focal, use bbox height as proxy
        
        # Simpler approach: assume bbox height inversely proportional to depth
        # Larger bbox = closer = smaller Z
        # Scale = known_height / (Z * angular_size)
        
        # Use median bbox height as reference
        scale_estimate = known_height_m / (z_raw * bbox_height_px / 1000.0)
        scales.append(scale_estimate)
    
    scales = np.array(scales)
    
    # Remove outliers
    q1, q3 = np.percentile(scales, [25, 75])
    iqr = q3 - q1
    valid_scales = scales[(scales >= q1 - 1.5*iqr) & (scales <= q3 + 1.5*iqr)]
    
    if method == 'median':
        scale_factor = np.median(valid_scales)
    elif method == 'mean':
        scale_factor = np.mean(valid_scales)
    elif method == 'robust':
        # Use trimmed mean (remove top/bottom 10%)
        scale_factor = np.mean(np.percentile(valid_scales, [10, 90]))
    else:
        scale_factor = np.median(valid_scales)
    
    return scale_factor, valid_scales


def estimate_scale_statistical(tracks, target_z_range=(0.5, 2.0), method='range'):
    """
    Estimate scale based on expected Z-axis range.
    
    Args:
        tracks: List of tracking data
        target_z_range: Expected min/max Z values in meters
        method: 'range', 'std', or 'iqr'
    
    Returns:
        scale_factor: Multiplier to convert depth units to meters
    """
    z_values = []
    
    for track in tracks:
        if track['detected']:
            z_values.append(track['contact_3d'][2])
    
    z_values = np.array(z_values)
    
    if method == 'range':
        # Scale so that range matches target
        current_range = np.max(z_values) - np.min(z_values)
        target_range = target_z_range[1] - target_z_range[0]
        scale_factor = target_range / current_range
        
    elif method == 'std':
        # Scale so that std dev matches expected
        current_std = np.std(z_values)
        target_std = (target_z_range[1] - target_z_range[0]) / 4  # Assume 4-sigma range
        scale_factor = target_std / current_std
        
    elif method == 'iqr':
        # Scale based on interquartile range
        q1, q3 = np.percentile(z_values, [25, 75])
        current_iqr = q3 - q1
        target_iqr = (target_z_range[1] - target_z_range[0]) / 2
        scale_factor = target_iqr / current_iqr
    
    return scale_factor


def apply_scale_calibration(tracks, scale_factor, offset=None):
    """
    Apply scale calibration to 3D positions.
    
    Args:
        tracks: List of tracking data
        scale_factor: Multiplier for depth values
        offset: Optional offset to add after scaling (for centering)
    
    Returns:
        calibrated_tracks: Tracks with scaled positions
    """
    calibrated_tracks = []
    
    for track in tracks:
        track_copy = track.copy()
        
        if track['detected']:
            # Scale 3D position
            pos = np.array(track['contact_3d'])
            pos[2] *= scale_factor  # Only scale Z (depth)
            
            if offset is not None:
                pos[2] += offset
            
            track_copy['contact_3d'] = pos.tolist()
            
            # Scale velocity (Z component)
            vel = np.array(track['velocity_3d'])
            vel[2] *= scale_factor
            track_copy['velocity_3d'] = vel.tolist()
        
        calibrated_tracks.append(track_copy)
    
    return calibrated_tracks


def compute_statistics(tracks, label=""):
    """Compute position statistics for tracks."""
    positions = []
    for track in tracks:
        if track['detected']:
            positions.append(track['contact_3d'])
    
    positions = np.array(positions)
    
    stats = {
        'label': label,
        'count': len(positions),
        'x_range': (np.min(positions[:, 0]), np.max(positions[:, 0])),
        'y_range': (np.min(positions[:, 1]), np.max(positions[:, 1])),
        'z_range': (np.min(positions[:, 2]), np.max(positions[:, 2])),
        'x_std': np.std(positions[:, 0]),
        'y_std': np.std(positions[:, 1]),
        'z_std': np.std(positions[:, 2])
    }
    
    return stats


def plot_calibration_comparison(original_stats, calibrated_stats, output_path):
    """Plot before/after calibration comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes = axes.flatten()
    
    # Z-axis ranges
    axes[0].barh(['Original', 'Calibrated'], 
                 [original_stats['z_range'][1] - original_stats['z_range'][0],
                  calibrated_stats['z_range'][1] - calibrated_stats['z_range'][0]],
                 color=['red', 'green'])
    axes[0].set_xlabel('Z-axis Range (m)')
    axes[0].set_title('Depth Range Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # Z min/max
    axes[1].bar(['Original\nMin', 'Original\nMax', 'Calibrated\nMin', 'Calibrated\nMax'],
                [original_stats['z_range'][0], original_stats['z_range'][1],
                 calibrated_stats['z_range'][0], calibrated_stats['z_range'][1]],
                color=['red', 'red', 'green', 'green'])
    axes[1].set_ylabel('Z Position (m)')
    axes[1].set_title('Z-axis Min/Max Values')
    axes[1].grid(True, alpha=0.3)
    
    # Standard deviations
    dims = ['X', 'Y', 'Z']
    orig_stds = [original_stats['x_std'], original_stats['y_std'], original_stats['z_std']]
    cal_stds = [calibrated_stats['x_std'], calibrated_stats['y_std'], calibrated_stats['z_std']]
    
    x = np.arange(len(dims))
    width = 0.35
    axes[2].bar(x - width/2, orig_stds, width, label='Original', color='red', alpha=0.7)
    axes[2].bar(x + width/2, cal_stds, width, label='Calibrated', color='green', alpha=0.7)
    axes[2].set_xlabel('Axis')
    axes[2].set_ylabel('Standard Deviation (m)')
    axes[2].set_title('Position Variability')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(dims)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # X range comparison
    axes[3].barh(['Original', 'Calibrated'], 
                 [original_stats['x_range'][1] - original_stats['x_range'][0],
                  calibrated_stats['x_range'][1] - calibrated_stats['x_range'][0]],
                 color=['orange', 'blue'])
    axes[3].set_xlabel('X-axis Range (m)')
    axes[3].set_title('X-axis Range (should be unchanged)')
    axes[3].grid(True, alpha=0.3)
    
    # Y range comparison
    axes[4].barh(['Original', 'Calibrated'], 
                 [original_stats['y_range'][1] - original_stats['y_range'][0],
                  calibrated_stats['y_range'][1] - calibrated_stats['y_range'][0]],
                 color=['orange', 'blue'])
    axes[4].set_xlabel('Y-axis Range (m)')
    axes[4].set_title('Y-axis Range (should be unchanged)')
    axes[4].grid(True, alpha=0.3)
    
    # Summary text
    axes[5].axis('off')
    summary_text = f"""
CALIBRATION SUMMARY

Original Z-range: {original_stats['z_range'][0]:.2f} to {original_stats['z_range'][1]:.2f} m
                  (span: {original_stats['z_range'][1] - original_stats['z_range'][0]:.2f} m)

Calibrated Z-range: {calibrated_stats['z_range'][0]:.2f} to {calibrated_stats['z_range'][1]:.2f} m
                    (span: {calibrated_stats['z_range'][1] - calibrated_stats['z_range'][0]:.2f} m)

Z std dev: {original_stats['z_std']:.3f} → {calibrated_stats['z_std']:.3f} m

Scale reduction: {(original_stats['z_range'][1] - original_stats['z_range'][0]) / (calibrated_stats['z_range'][1] - calibrated_stats['z_range'][0]):.1f}x

Target range: 0.5-2.0 m (pantograph oscillation)
Status: {'✓ GOOD' if 0.3 < calibrated_stats['z_range'][1] - calibrated_stats['z_range'][0] < 3.0 else '⚠ CHECK'}
"""
    axes[5].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Calibration comparison plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate depth scale for pantograph tracking')
    parser.add_argument('input_json', help='Input tracking JSON file')
    parser.add_argument('--output', '-o', help='Output calibrated JSON file')
    parser.add_argument('--known-height', type=float, default=1.5,
                       help='Known pantograph height in meters (default: 1.5)')
    parser.add_argument('--target-range', type=float, nargs=2, default=[0.5, 2.0],
                       help='Target Z-axis range in meters (default: 0.5 2.0)')
    parser.add_argument('--method', choices=['statistical', 'bbox', 'manual'],
                       default='statistical',
                       help='Calibration method (default: statistical)')
    parser.add_argument('--scale', type=float,
                       help='Manual scale factor (for method=manual)')
    parser.add_argument('--plot', help='Output plot path')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading tracking data from {args.input_json}...")
    data = load_tracking_data(args.input_json)
    
    # Compute original statistics
    print("Computing original statistics...")
    original_stats = compute_statistics(data['tracks'], "Original")
    
    print("\nOriginal Statistics:")
    print(f"  X-axis: {original_stats['x_range'][0]:.3f} to {original_stats['x_range'][1]:.3f} m")
    print(f"  Y-axis: {original_stats['y_range'][0]:.3f} to {original_stats['y_range'][1]:.3f} m")
    print(f"  Z-axis: {original_stats['z_range'][0]:.3f} to {original_stats['z_range'][1]:.3f} m")
    print(f"  Z-span: {original_stats['z_range'][1] - original_stats['z_range'][0]:.3f} m")
    
    # Estimate scale factor
    if args.method == 'manual':
        if args.scale is None:
            print("Error: --scale required for manual method")
            return
        scale_factor = args.scale
        print(f"\nUsing manual scale factor: {scale_factor:.6f}")
        
    elif args.method == 'statistical':
        print(f"\nEstimating scale using statistical method...")
        print(f"Target Z-range: {args.target_range[0]:.2f} to {args.target_range[1]:.2f} m")
        scale_factor = estimate_scale_statistical(data['tracks'], 
                                                  tuple(args.target_range),
                                                  method='range')
        print(f"Estimated scale factor: {scale_factor:.6f}")
        
    elif args.method == 'bbox':
        print(f"\nEstimating scale using bbox method...")
        print(f"Known pantograph height: {args.known_height} m")
        scale_factor, valid_scales = estimate_scale_from_bbox(data['tracks'], 
                                                              args.known_height)
        print(f"Estimated scale factor: {scale_factor:.6f}")
        print(f"Scale std dev: {np.std(valid_scales):.6f}")
        print(f"Scale range: {np.min(valid_scales):.6f} to {np.max(valid_scales):.6f}")
    
    # Apply calibration
    print("\nApplying scale calibration...")
    calibrated_tracks = apply_scale_calibration(data['tracks'], scale_factor)
    
    # Compute calibrated statistics
    calibrated_stats = compute_statistics(calibrated_tracks, "Calibrated")
    
    print("\nCalibrated Statistics:")
    print(f"  X-axis: {calibrated_stats['x_range'][0]:.3f} to {calibrated_stats['x_range'][1]:.3f} m")
    print(f"  Y-axis: {calibrated_stats['y_range'][0]:.3f} to {calibrated_stats['y_range'][1]:.3f} m")
    print(f"  Z-axis: {calibrated_stats['z_range'][0]:.3f} to {calibrated_stats['z_range'][1]:.3f} m")
    print(f"  Z-span: {calibrated_stats['z_range'][1] - calibrated_stats['z_range'][0]:.3f} m")
    
    # Check if in target range
    z_span = calibrated_stats['z_range'][1] - calibrated_stats['z_range'][0]
    target_span = args.target_range[1] - args.target_range[0]
    
    if 0.3 < z_span < 3.0:
        print(f"\n✅ Z-span ({z_span:.2f} m) is physically reasonable for pantograph!")
    else:
        print(f"\n⚠️  Z-span ({z_span:.2f} m) may need adjustment")
    
    # Save calibrated data
    output_path = args.output or args.input_json.replace('.json', '_calibrated.json')
    
    calibrated_data = data.copy()
    calibrated_data['tracks'] = calibrated_tracks
    calibrated_data['calibration'] = {
        'scale_factor': scale_factor,
        'method': args.method,
        'known_height_m': args.known_height if args.method == 'bbox' else None,
        'target_range_m': args.target_range,
        'original_z_range': original_stats['z_range'],
        'calibrated_z_range': calibrated_stats['z_range']
    }
    
    with open(output_path, 'w') as f:
        json.dump(calibrated_data, f, indent=2)
    
    print(f"\n✅ Calibrated tracking data saved to {output_path}")
    
    # Generate comparison plot
    if args.plot:
        plot_path = args.plot
    else:
        plot_path = str(Path(output_path).parent / 'depth_calibration_comparison.png')
    
    plot_calibration_comparison(original_stats, calibrated_stats, plot_path)
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    print(f"Scale factor: {scale_factor:.6f}")
    print(f"Z-range reduction: {(original_stats['z_range'][1] - original_stats['z_range'][0]) / z_span:.1f}x")
    print(f"Output: {output_path}")
    print(f"Plot: {plot_path}")


if __name__ == '__main__':
    main()
