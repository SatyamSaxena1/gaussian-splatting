#!/usr/bin/env python3
"""Analyze contact tracking results and generate statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_json", type=Path, help="Contact tracking JSON file")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    
    if not args.track_json.exists():
        print(f"[analyze_tracking] File not found: {args.track_json}", file=sys.stderr)
        return 1
    
    with open(args.track_json) as f:
        data = json.load(f)
    
    tracks = data["tracks"]
    metadata = data["metadata"]
    
    print("=" * 60)
    print("PANTOGRAPH CONTACT TRACKING ANALYSIS")
    print("=" * 60)
    
    # Detection statistics
    print(f"\n📊 DETECTION STATISTICS:")
    print(f"  Total frames:        {metadata['total_frames']}")
    print(f"  Detected frames:     {metadata['detected_frames']}")
    print(f"  Detection rate:      {metadata['detection_rate']*100:.1f}%")
    print(f"  Average confidence:  {metadata['avg_confidence']:.3f}")
    print(f"  Model:               {metadata['model']}")
    
    # Extract detected tracks
    detected = [t for t in tracks if t["detected"]]
    if not detected:
        print("\n⚠️  No detections found!")
        return 0
    
    confidences = [t["confidence"] for t in detected]
    print(f"\n📈 CONFIDENCE DISTRIBUTION:")
    print(f"  Min confidence:      {min(confidences):.3f}")
    print(f"  Max confidence:      {max(confidences):.3f}")
    print(f"  Mean confidence:     {np.mean(confidences):.3f}")
    print(f"  Std confidence:      {np.std(confidences):.3f}")
    
    # 3D position analysis
    positions_3d = np.array([t["contact_3d"] for t in detected])
    
    print(f"\n🎯 3D CONTACT POINT STATISTICS:")
    print(f"  X range: [{positions_3d[:, 0].min():.3f}, {positions_3d[:, 0].max():.3f}]")
    print(f"  Y range: [{positions_3d[:, 1].min():.3f}, {positions_3d[:, 1].max():.3f}]")
    print(f"  Z range: [{positions_3d[:, 2].min():.3f}, {positions_3d[:, 2].max():.3f}]")
    print(f"  Mean position: ({positions_3d[:, 0].mean():.3f}, {positions_3d[:, 1].mean():.3f}, {positions_3d[:, 2].mean():.3f})")
    
    # Velocity analysis
    velocities = [t.get("velocity_3d", [0, 0, 0]) for t in detected if "velocity_3d" in t]
    if velocities:
        velocities = np.array(velocities)
        speeds = np.linalg.norm(velocities, axis=1)
        
        print(f"\n⚡ VELOCITY STATISTICS:")
        print(f"  Mean speed:          {speeds.mean():.4f}")
        print(f"  Max speed:           {speeds.max():.4f}")
        print(f"  Std speed:           {speeds.std():.4f}")
    
    # Bounding box size analysis
    bboxes = [t["bbox"] for t in detected]
    widths = [b[2] - b[0] for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]
    
    print(f"\n📦 BOUNDING BOX STATISTICS:")
    print(f"  Mean width:          {np.mean(widths):.1f} px")
    print(f"  Mean height:         {np.mean(heights):.1f} px")
    print(f"  Width range:         [{min(widths)}, {max(widths)}] px")
    print(f"  Height range:        [{min(heights)}, {max(heights)}] px")
    
    # Detection gaps
    gaps = []
    current_gap = 0
    for t in tracks:
        if t["detected"]:
            if current_gap > 0:
                gaps.append(current_gap)
            current_gap = 0
        else:
            current_gap += 1
    
    if gaps:
        print(f"\n🔍 DETECTION GAPS:")
        print(f"  Number of gaps:      {len(gaps)}")
        print(f"  Mean gap length:     {np.mean(gaps):.1f} frames")
        print(f"  Max gap length:      {max(gaps)} frames")
    
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
