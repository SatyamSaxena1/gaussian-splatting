#!/usr/bin/env python3
"""Export YOLO-tracked contact points to USD with confidence visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
except ImportError:
    print("Error: USD Python module not found. Install with: pip install usd-core", file=sys.stderr)
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_json", type=Path, help="Contact tracking JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output USD file")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for animation")
    parser.add_argument("--sphere-radius", type=float, default=0.05, help="Contact marker radius")
    parser.add_argument("--velocity-scale", type=float, default=1.0, help="Velocity vector scale")
    return parser.parse_args()


def _confidence_to_color(conf: float) -> tuple[float, float, float]:
    """Map confidence [0, 1] to color: red (low) -> yellow -> green (high)."""
    if conf < 0.5:
        # Red to yellow
        r = 1.0
        g = conf * 2.0
        b = 0.0
    else:
        # Yellow to green
        r = 1.0 - (conf - 0.5) * 2.0
        g = 1.0
        b = 0.0
    return (r, g, b)


def main() -> int:
    args = _parse_args()
    
    if not args.track_json.exists():
        print(f"[export_yolo_track_to_usd] Track JSON not found: {args.track_json}", file=sys.stderr)
        return 1
    
    # Load tracking data
    with open(args.track_json) as f:
        data = json.load(f)
    
    tracks = data["tracks"]
    metadata = data["metadata"]
    fps = args.fps
    
    output_usd = args.output or args.track_json.parent / "contact_track_yolo.usda"
    
    print(f"[export_yolo_track_to_usd] Creating USD: {output_usd}")
    print(f"[export_yolo_track_to_usd] Frames: {len(tracks)}")
    print(f"[export_yolo_track_to_usd] Detection rate: {metadata['detection_rate']*100:.1f}%")
    
    # Create USD stage
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.SetMetadata("timeCodesPerSecond", fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(tracks) - 1)
    
    # Create root transform
    root = UsdGeom.Xform.Define(stage, "/ContactTrack")
    
    # Create contact point sphere
    sphere_path = "/ContactTrack/ContactPoint"
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.GetRadiusAttr().Set(args.sphere_radius)
    
    # Prepare animation data
    positions = []
    confidences = []
    detected_flags = []
    
    for track in tracks:
        if track["detected"]:
            pos = track["contact_3d"]
            positions.append(Gf.Vec3f(pos[0], pos[1], pos[2]))
            confidences.append(track["confidence"])
            detected_flags.append(True)
        else:
            # Hide by moving far away
            positions.append(Gf.Vec3f(0, 0, -1000))
            confidences.append(0.0)
            detected_flags.append(False)
    
    # Set position animation
    translate_op = sphere.AddTranslateOp()
    for frame_idx, pos in enumerate(positions):
        translate_op.Set(pos, frame_idx)
    
    # Set color animation based on confidence
    color_attr = sphere.GetDisplayColorAttr()
    if not color_attr:
        color_attr = sphere.CreateDisplayColorAttr()
    
    for frame_idx, (conf, detected) in enumerate(zip(confidences, detected_flags)):
        if detected:
            color = _confidence_to_color(conf)
        else:
            color = (0.5, 0.5, 0.5)  # Gray for undetected
        color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*color)]), frame_idx)
    
    # Create velocity vectors (as line segments)
    lines_path = "/ContactTrack/VelocityVectors"
    lines_geom = UsdGeom.BasisCurves.Define(stage, lines_path)
    lines_geom.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    lines_geom.CreateWidthsAttr().Set([2.0])
    
    # Build velocity line segments
    all_points = []
    all_counts = []
    
    for frame_idx, track in enumerate(tracks):
        if track.get("velocity_3d") and track["detected"]:
            pos = np.array(track["contact_3d"])
            vel = np.array(track["velocity_3d"])
            
            # Scale velocity for visibility
            vel_end = pos + vel * args.velocity_scale
            
            all_points.extend([
                Gf.Vec3f(pos[0], pos[1], pos[2]),
                Gf.Vec3f(vel_end[0], vel_end[1], vel_end[2])
            ])
            all_counts.append(2)
    
    if all_points:
        lines_geom.GetPointsAttr().Set(Vt.Vec3fArray(all_points))
        lines_geom.GetCurveVertexCountsAttr().Set(Vt.IntArray(all_counts))
        lines_geom.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1, 1, 0)]))  # Yellow
    
    # Add metadata
    root.GetPrim().SetCustomDataByKey("tracking_metadata", metadata)
    root.GetPrim().SetCustomDataByKey("total_frames", len(tracks))
    root.GetPrim().SetCustomDataByKey("detection_rate", metadata["detection_rate"])
    root.GetPrim().SetCustomDataByKey("avg_confidence", metadata["avg_confidence"])
    
    stage.Save()
    
    print(f"[export_yolo_track_to_usd] USD created successfully")
    print(f"[export_yolo_track_to_usd] Animation: {len(tracks)} frames @ {fps} fps")
    print(f"\nView with: usdview {output_usd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
