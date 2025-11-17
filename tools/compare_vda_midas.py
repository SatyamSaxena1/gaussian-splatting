#!/usr/bin/env python3
"""
Compare VDA vs MiDaS temporal consistency.
Measures depth jitter/tortuosity between frames.
"""

import json
import numpy as np
import argparse
from pathlib import Path
import cv2
from typing import List, Tuple

def load_tracking_data(file_path: Path) -> dict:
    """Load tracking JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_tortuosity(positions_3d: List[Tuple[float, float, float]]) -> float:
    """
    Calculate tortuosity metric: ratio of path length to straight-line distance.
    Higher values indicate more jitter/noise.
    """
    if len(positions_3d) < 2:
        return 0.0
    
    # Calculate total path length (sum of consecutive distances)
    path_length = 0.0
    for i in range(len(positions_3d) - 1):
        p1 = np.array(positions_3d[i])
        p2 = np.array(positions_3d[i + 1])
        path_length += np.linalg.norm(p2 - p1)
    
    # Calculate straight-line distance (euclidean distance between start and end)
    start = np.array(positions_3d[0])
    end = np.array(positions_3d[-1])
    straight_distance = np.linalg.norm(end - start)
    
    # Tortuosity = path_length / straight_distance
    # For a perfectly straight line: tortuosity = 1.0
    # Higher values indicate more jitter
    if straight_distance < 1e-6:
        return 0.0
    
    return path_length / straight_distance

def calculate_depth_variance(depths: List[float]) -> float:
    """Calculate variance of depth values."""
    if len(depths) < 2:
        return 0.0
    return float(np.var(depths))

def calculate_depth_change_rate(depths: List[float]) -> float:
    """Calculate average frame-to-frame depth change."""
    if len(depths) < 2:
        return 0.0
    
    changes = []
    for i in range(len(depths) - 1):
        changes.append(abs(depths[i+1] - depths[i]))
    
    return float(np.mean(changes))

def analyze_tracking(tracking_data: dict, method_name: str) -> dict:
    """Analyze temporal consistency of tracking data."""
    
    tortuosities = []
    depth_variances = []
    depth_change_rates = []
    track_lengths = []
    
    for track in tracking_data:
        if 'positions_3d' not in track or len(track['positions_3d']) < 3:
            continue
        
        positions_3d = track['positions_3d']
        depths = [pos[2] for pos in positions_3d]  # Z coordinate
        
        # Calculate metrics
        tort = calculate_tortuosity(positions_3d)
        var = calculate_depth_variance(depths)
        change_rate = calculate_depth_change_rate(depths)
        
        tortuosities.append(tort)
        depth_variances.append(var)
        depth_change_rates.append(change_rate)
        track_lengths.append(len(positions_3d))
    
    if not tortuosities:
        return {
            'method': method_name,
            'error': 'No valid tracks found'
        }
    
    return {
        'method': method_name,
        'num_tracks': len(tortuosities),
        'avg_track_length': float(np.mean(track_lengths)),
        'tortuosity': {
            'mean': float(np.mean(tortuosities)),
            'median': float(np.median(tortuosities)),
            'std': float(np.std(tortuosities)),
            'min': float(np.min(tortuosities)),
            'max': float(np.max(tortuosities))
        },
        'depth_variance': {
            'mean': float(np.mean(depth_variances)),
            'median': float(np.median(depth_variances)),
            'std': float(np.std(depth_variances))
        },
        'depth_change_rate': {
            'mean': float(np.mean(depth_change_rates)),
            'median': float(np.median(depth_change_rates)),
            'std': float(np.std(depth_change_rates))
        }
    }

def print_comparison(vda_stats: dict, midas_stats: dict = None):
    """Print comparison results."""
    
    print("\n" + "="*60)
    print("TEMPORAL CONSISTENCY COMPARISON")
    print("="*60)
    
    # VDA results
    print(f"\n{'='*60}")
    print(f"VDA (Video Depth Anything)")
    print(f"{'='*60}")
    print(f"Tracks analyzed: {vda_stats['num_tracks']}")
    print(f"Avg track length: {vda_stats['avg_track_length']:.1f} frames")
    print(f"\nTortuosity (lower = smoother):")
    print(f"  Mean:   {vda_stats['tortuosity']['mean']:.3f}")
    print(f"  Median: {vda_stats['tortuosity']['median']:.3f}")
    print(f"  Std:    {vda_stats['tortuosity']['std']:.3f}")
    print(f"  Range:  [{vda_stats['tortuosity']['min']:.3f}, {vda_stats['tortuosity']['max']:.3f}]")
    print(f"\nDepth Variance:")
    print(f"  Mean:   {vda_stats['depth_variance']['mean']:.3f}")
    print(f"  Median: {vda_stats['depth_variance']['median']:.3f}")
    print(f"\nDepth Change Rate (m/frame):")
    print(f"  Mean:   {vda_stats['depth_change_rate']['mean']:.3f}")
    print(f"  Median: {vda_stats['depth_change_rate']['median']:.3f}")
    
    # MiDaS results (if available)
    if midas_stats and 'error' not in midas_stats:
        print(f"\n{'='*60}")
        print(f"MiDaS")
        print(f"{'='*60}")
        print(f"Tracks analyzed: {midas_stats['num_tracks']}")
        print(f"Avg track length: {midas_stats['avg_track_length']:.1f} frames")
        print(f"\nTortuosity (lower = smoother):")
        print(f"  Mean:   {midas_stats['tortuosity']['mean']:.3f}")
        print(f"  Median: {midas_stats['tortuosity']['median']:.3f}")
        print(f"  Std:    {midas_stats['tortuosity']['std']:.3f}")
        print(f"  Range:  [{midas_stats['tortuosity']['min']:.3f}, {midas_stats['tortuosity']['max']:.3f}]")
        print(f"\nDepth Variance:")
        print(f"  Mean:   {midas_stats['depth_variance']['mean']:.3f}")
        print(f"  Median: {midas_stats['depth_variance']['median']:.3f}")
        print(f"\nDepth Change Rate (m/frame):")
        print(f"  Mean:   {midas_stats['depth_change_rate']['mean']:.3f}")
        print(f"  Median: {midas_stats['depth_change_rate']['median']:.3f}")
        
        # Comparison
        print(f"\n{'='*60}")
        print(f"COMPARISON (VDA vs MiDaS)")
        print(f"{'='*60}")
        
        vda_tort = vda_stats['tortuosity']['mean']
        midas_tort = midas_stats['tortuosity']['mean']
        improvement = ((midas_tort - vda_tort) / midas_tort) * 100
        
        print(f"\nTortuosity Improvement: {improvement:+.1f}%")
        if improvement > 0:
            print(f"  ✓ VDA is {improvement:.1f}% smoother than MiDaS")
        else:
            print(f"  ✗ VDA is {abs(improvement):.1f}% noisier than MiDaS")
        
        vda_var = vda_stats['depth_variance']['mean']
        midas_var = midas_stats['depth_variance']['mean']
        var_improvement = ((midas_var - vda_var) / midas_var) * 100
        
        print(f"\nDepth Variance Reduction: {var_improvement:+.1f}%")
        if var_improvement > 0:
            print(f"  ✓ VDA has {var_improvement:.1f}% less depth variance")
        else:
            print(f"  ✗ VDA has {abs(var_improvement):.1f}% more depth variance")
    
    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Compare VDA vs MiDaS temporal consistency')
    parser.add_argument('--vda', type=str, required=True, help='VDA tracking JSON file')
    parser.add_argument('--midas', type=str, help='MiDaS tracking JSON file (optional)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Load VDA data
    print(f"Loading VDA tracking: {args.vda}")
    vda_data = load_tracking_data(Path(args.vda))
    
    # Analyze VDA
    print("Analyzing VDA temporal consistency...")
    vda_stats = analyze_tracking(vda_data, 'VDA')
    
    # Load and analyze MiDaS if available
    midas_stats = None
    if args.midas:
        print(f"\nLoading MiDaS tracking: {args.midas}")
        midas_data = load_tracking_data(Path(args.midas))
        print("Analyzing MiDaS temporal consistency...")
        midas_stats = analyze_tracking(midas_data, 'MiDaS')
    
    # Print comparison
    print_comparison(vda_stats, midas_stats)
    
    # Save results
    if args.output:
        results = {
            'vda': vda_stats,
            'midas': midas_stats
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    main()
