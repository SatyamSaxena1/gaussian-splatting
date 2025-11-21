#!/usr/bin/env python3
"""
Kalman Filter for Trajectory Smoothing

Applies Kalman filtering to reduce noise in 3D tracking trajectories.
Handles missing detections and provides optimal state estimation.
"""

import json
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


class KalmanFilter3D:
    """3D Kalman filter for position and velocity estimation."""
    
    def __init__(self, process_noise=0.1, measurement_noise=1.0, dt=1.0):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise covariance (model uncertainty)
            measurement_noise: Measurement noise covariance (sensor noise)
            dt: Time step between measurements
        """
        # State: [x, y, z, vx, vy, vz]
        self.state_dim = 6
        self.measurement_dim = 3
        
        # State vector
        self.x = np.zeros(self.state_dim)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        # Higher noise for velocity (more uncertain)
        q_pos = process_noise
        q_vel = process_noise * 2
        self.Q = block_diag(
            np.eye(3) * q_pos,
            np.eye(3) * q_vel
        )
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # State covariance
        self.P = np.eye(self.state_dim) * 10.0
        
        # Initialization flag
        self.initialized = False
    
    def initialize(self, measurement):
        """Initialize filter with first measurement."""
        self.x[:3] = measurement
        self.x[3:] = 0  # Zero initial velocity
        self.initialized = True
    
    def predict(self):
        """Predict next state."""
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:3].copy()
    
    def update(self, measurement, R=None):
        """
        Update state with measurement.
        
        Args:
            measurement: Measurement vector [x, y, z]
            R: Optional custom measurement noise covariance for this step
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.x
        
        # Use custom R if provided, else default
        R_curr = R if R is not None else self.R
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R_curr
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:3].copy()
    
    def get_position(self):
        """Get current position estimate."""
        return self.x[:3].copy()
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.x[3:].copy()
    
    def get_covariance(self):
        """Get position covariance."""
        return self.P[:3, :3].copy()


def load_tracking_data(json_path):
    """Load tracking JSON data."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def apply_kalman_filter(tracks, process_noise=0.1, measurement_noise=1.0, fps=30.0):
    """
    Apply Kalman filter to tracking data.
    
    Args:
        tracks: List of tracking data
        process_noise: Process noise parameter
        measurement_noise: Measurement noise parameter (base value)
        fps: Frame rate for dt calculation
    
    Returns:
        filtered_tracks: Tracks with smoothed positions
        covariances: Position covariances for each frame
    """
    dt = 1.0 / fps
    kf = KalmanFilter3D(process_noise=process_noise, 
                        measurement_noise=measurement_noise,
                        dt=dt)
    
    filtered_tracks = []
    covariances = []
    
    for i, track in enumerate(tracks):
        track_copy = track.copy()
        
        if track['detected']:
            measurement = np.array(track['contact_3d'])
            
            # Dynamic Measurement Noise
            # If we have depth_std, use it to scale the Z-axis noise
            # Base noise is measurement_noise (e.g. 1.0)
            # If depth_std is high, we trust the measurement LESS (higher R)
            
            R_dynamic = None
            depth_std = track.get('depth_std', 0.0)
            
            if depth_std > 0:
                # Scale factor: if std is 0.1 (10cm), variance is 0.01
                # We want to scale the Z-noise component
                # Heuristic: R_z = base_noise + (scale * depth_std)^2
                # Let's say base noise covers XY error (pixel error).
                # Z error is dominated by depth estimation error.
                
                base_R = np.eye(3) * measurement_noise
                
                # Boost Z noise based on depth_std
                # We multiply by a factor (e.g. 100) because depth_std is in meters/units
                # and we want to be conservative when it's noisy.
                z_noise_variance = measurement_noise + (depth_std * 10.0)**2
                
                base_R[2, 2] = z_noise_variance
                R_dynamic = base_R
            
            if not kf.initialized:
                # Initialize with first measurement
                kf.initialize(measurement)
                filtered_pos = kf.get_position()
                filtered_vel = kf.get_velocity()
            else:
                # Predict then update
                kf.predict()
                filtered_pos = kf.update(measurement, R=R_dynamic)
                filtered_vel = kf.get_velocity()
            
            track_copy['contact_3d'] = filtered_pos.tolist()
            track_copy['velocity_3d'] = filtered_vel.tolist()
            track_copy['contact_3d_raw'] = track['contact_3d']  # Keep original
            track_copy['filtered'] = True
            
            cov = kf.get_covariance()
            covariances.append(cov)
            
        else:
            # Missing detection - use prediction only
            if kf.initialized:
                predicted_pos = kf.predict()
                predicted_vel = kf.get_velocity()
                
                track_copy['contact_3d'] = predicted_pos.tolist()
                track_copy['velocity_3d'] = predicted_vel.tolist()
                track_copy['detected'] = False  # Mark as interpolated
                track_copy['interpolated'] = True
                track_copy['filtered'] = True
                
                cov = kf.get_covariance()
                covariances.append(cov)
            else:
                # Can't predict without initialization
                covariances.append(np.eye(3) * np.inf)
        
        filtered_tracks.append(track_copy)
    
    return filtered_tracks, covariances


def compute_filtering_metrics(original_tracks, filtered_tracks):
    """Compute metrics to assess filtering quality."""
    original_positions = []
    filtered_positions = []
    
    for orig, filt in zip(original_tracks, filtered_tracks):
        if orig['detected']:
            original_positions.append(orig['contact_3d'])
            filtered_positions.append(filt['contact_3d'])
    
    original_positions = np.array(original_positions)
    filtered_positions = np.array(filtered_positions)
    
    # Compute differences
    differences = filtered_positions - original_positions
    rmse = np.sqrt(np.mean(differences**2, axis=0))
    
    # Compute smoothness (acceleration magnitude)
    def compute_smoothness(positions):
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        if len(accelerations) == 0:
            return 0.0
        return np.mean(np.linalg.norm(accelerations, axis=1))
    
    original_smoothness = compute_smoothness(original_positions)
    filtered_smoothness = compute_smoothness(filtered_positions)
    
    # Compute path lengths
    def compute_path_length(positions):
        segments = np.diff(positions, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return np.sum(lengths)
    
    original_path = compute_path_length(original_positions)
    filtered_path = compute_path_length(filtered_positions)
    
    metrics = {
        'rmse': rmse,
        'mean_rmse': np.mean(rmse),
        'max_rmse': np.max(rmse),
        'original_smoothness': original_smoothness,
        'filtered_smoothness': filtered_smoothness,
        'smoothness_improvement': (original_smoothness - filtered_smoothness) / original_smoothness,
        'original_path_length': original_path,
        'filtered_path_length': filtered_path,
        'path_reduction': (original_path - filtered_path) / original_path
    }
    
    return metrics


def plot_filtering_results(original_tracks, filtered_tracks, covariances, output_path):
    """Plot filtering results comparison."""
    # Extract positions
    original_positions = []
    filtered_positions = []
    frames = []
    uncertainties_list = []
    
    for i, (orig, filt) in enumerate(zip(original_tracks, filtered_tracks)):
        if orig['detected']:
            original_positions.append(orig['contact_3d'])
            filtered_positions.append(filt['contact_3d'])
            frames.append(i)
            # Only add uncertainty for detected frames
            if i < len(covariances):
                uncertainties_list.append(np.sqrt(np.diag(covariances[i])))
            else:
                uncertainties_list.append(np.zeros(3))
    
    original_positions = np.array(original_positions)
    filtered_positions = np.array(filtered_positions)
    frames = np.array(frames)
    time_s = frames / 30.0
    
    # Extract uncertainties (only for valid frames)
    uncertainties = np.array(uncertainties_list)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Position comparisons for each axis
    for i, (ax, label) in enumerate(zip(axes[:, 0], ['X', 'Y', 'Z'])):
        ax.plot(time_s, original_positions[:, i], 'r-', alpha=0.5, 
               linewidth=1, label='Original')
        ax.plot(time_s, filtered_positions[:, i], 'b-', alpha=0.8, 
               linewidth=1.5, label='Filtered')
        
        # Plot uncertainty bounds
        ax.fill_between(time_s,
                        filtered_positions[:, i] - uncertainties[:, i],
                        filtered_positions[:, i] + uncertainties[:, i],
                        alpha=0.2, color='blue', label='±1σ')
        
        ax.set_ylabel(f'{label} Position (m)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{label}-Position: Original vs Filtered')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Residuals for each axis
    residuals = filtered_positions - original_positions
    for i, (ax, label) in enumerate(zip(axes[:, 1], ['X', 'Y', 'Z'])):
        ax.plot(time_s, residuals[:, i], 'g-', alpha=0.7, linewidth=1)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel(f'{label} Residual (m)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{label} Filtering Residuals')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Filtering comparison plot saved to {output_path}")


def plot_3d_comparison(original_tracks, filtered_tracks, output_path):
    """Plot 3D trajectory comparison."""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract positions
    original_positions = []
    filtered_positions = []
    
    for orig, filt in zip(original_tracks, filtered_tracks):
        if orig['detected']:
            original_positions.append(orig['contact_3d'])
            filtered_positions.append(filt['contact_3d'])
    
    original_positions = np.array(original_positions)
    filtered_positions = np.array(filtered_positions)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(original_positions[:, 0], original_positions[:, 1], original_positions[:, 2],
            'r-', alpha=0.5, linewidth=1, label='Original')
    ax1.plot(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2],
            'b-', alpha=0.8, linewidth=1.5, label='Filtered')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # XY plane
    ax2 = fig.add_subplot(132)
    ax2.plot(original_positions[:, 0], original_positions[:, 1], 
            'r-', alpha=0.5, linewidth=1, label='Original')
    ax2.plot(filtered_positions[:, 0], filtered_positions[:, 1], 
            'b-', alpha=0.8, linewidth=1.5, label='Filtered')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ plane
    ax3 = fig.add_subplot(133)
    ax3.plot(original_positions[:, 0], original_positions[:, 2], 
            'r-', alpha=0.5, linewidth=1, label='Original')
    ax3.plot(filtered_positions[:, 0], filtered_positions[:, 2], 
            'b-', alpha=0.8, linewidth=1.5, label='Filtered')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Plane')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 3D comparison plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Apply Kalman filter to trajectory data')
    parser.add_argument('input_json', help='Input tracking JSON file')
    parser.add_argument('--output', '-o', help='Output filtered JSON file')
    parser.add_argument('--process-noise', type=float, default=0.1,
                       help='Process noise parameter (default: 0.1)')
    parser.add_argument('--measurement-noise', type=float, default=1.0,
                       help='Measurement noise parameter (default: 1.0)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Video frame rate (default: 30.0)')
    parser.add_argument('--plot-dir', help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading tracking data from {args.input_json}...")
    data = load_tracking_data(args.input_json)
    
    print(f"Total frames: {len(data['tracks'])}")
    detected_frames = sum(1 for t in data['tracks'] if t['detected'])
    print(f"Detected frames: {detected_frames}")
    
    # Apply Kalman filter
    print(f"\nApplying Kalman filter...")
    print(f"  Process noise: {args.process_noise}")
    print(f"  Measurement noise: {args.measurement_noise}")
    print(f"  Frame rate: {args.fps} fps")
    
    filtered_tracks, covariances = apply_kalman_filter(
        data['tracks'],
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
        fps=args.fps
    )
    
    # Compute metrics
    print("\nComputing filtering metrics...")
    metrics = compute_filtering_metrics(data['tracks'], filtered_tracks)
    
    print("\nFiltering Results:")
    print(f"  RMSE (X, Y, Z): {metrics['rmse'][0]:.4f}, {metrics['rmse'][1]:.4f}, {metrics['rmse'][2]:.4f} m")
    print(f"  Mean RMSE: {metrics['mean_rmse']:.4f} m")
    print(f"  Max RMSE: {metrics['max_rmse']:.4f} m")
    print(f"  Smoothness improvement: {metrics['smoothness_improvement']*100:.1f}%")
    print(f"  Path length reduction: {metrics['path_reduction']*100:.1f}%")
    print(f"  Original path: {metrics['original_path_length']:.2f} m")
    print(f"  Filtered path: {metrics['filtered_path_length']:.2f} m")
    
    # Save filtered data
    output_path = args.output or args.input_json.replace('.json', '_filtered.json')
    
    filtered_data = data.copy()
    filtered_data['tracks'] = filtered_tracks
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {
        'rmse': metrics['rmse'].tolist(),
        'mean_rmse': float(metrics['mean_rmse']),
        'max_rmse': float(metrics['max_rmse']),
        'original_smoothness': float(metrics['original_smoothness']),
        'filtered_smoothness': float(metrics['filtered_smoothness']),
        'smoothness_improvement': float(metrics['smoothness_improvement']),
        'original_path_length': float(metrics['original_path_length']),
        'filtered_path_length': float(metrics['filtered_path_length']),
        'path_reduction': float(metrics['path_reduction'])
    }
    
    filtered_data['kalman_filter'] = {
        'process_noise': args.process_noise,
        'measurement_noise': args.measurement_noise,
        'fps': args.fps,
        'metrics': metrics_json
    }
    
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n✅ Filtered tracking data saved to {output_path}")
    
    # Generate plots
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
    else:
        plot_dir = Path(output_path).parent / 'kalman_filter_plots'
    
    plot_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating plots...")
    plot_filtering_results(data['tracks'], filtered_tracks, covariances,
                          plot_dir / 'kalman_filter_comparison.png')
    plot_3d_comparison(data['tracks'], filtered_tracks,
                      plot_dir / 'kalman_3d_trajectories.png')
    
    print("\n" + "="*70)
    print("KALMAN FILTERING COMPLETE")
    print("="*70)
    print(f"Smoothness improved by: {metrics['smoothness_improvement']*100:.1f}%")
    print(f"Path length reduced by: {metrics['path_reduction']*100:.1f}%")
    print(f"Output: {output_path}")
    print(f"Plots: {plot_dir}/")


if __name__ == '__main__':
    main()
