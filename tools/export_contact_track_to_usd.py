#!/usr/bin/env python3
"""Export contact tracking data to USD as animated spheres overlaid on static scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from pxr import Gf, Sdf, Usd, UsdGeom
except ImportError:
    print("[export_contact_track_to_usd] pxr module not available; ensure USD env is active", file=sys.stderr)
    sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True, help="Contact track JSON file")
    parser.add_argument("--static-stage", type=Path, required=True, help="Static scene USD stage")
    parser.add_argument("--output", type=Path, required=True, help="Output USD stage with contact track overlay")
    parser.add_argument("--marker-radius", type=float, default=0.05, help="Contact marker sphere radius")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for time sampling")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    
    if not args.track.exists():
        print(f"[export_contact_track_to_usd] Track file not found: {args.track}", file=sys.stderr)
        return 1
    
    if not args.static_stage.exists():
        print(f"[export_contact_track_to_usd] Static stage not found: {args.static_stage}", file=sys.stderr)
        return 1
    
    with open(args.track, "r") as f:
        track_data = json.load(f)
    
    frames = track_data["frames"]
    if not frames:
        print("[export_contact_track_to_usd] No tracked frames in input", file=sys.stderr)
        return 1
    
    # Create new stage
    stage = Usd.Stage.CreateNew(str(args.output))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    stage.SetTimeCodesPerSecond(args.fps)
    stage.SetFramesPerSecond(args.fps)
    
    # Reference the static scene
    static_ref = stage.DefinePrim("/StaticScene")
    static_ref.GetReferences().AddReference(str(args.static_stage))
    
    # Create contact marker hierarchy
    contact_root = UsdGeom.Xform.Define(stage, "/ContactTrack")
    contact_marker = UsdGeom.Sphere.Define(stage, "/ContactTrack/Marker")
    contact_marker.GetRadiusAttr().Set(args.marker_radius)
    contact_marker.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red marker
    
    # Animate the marker position
    xform = UsdGeom.Xformable(contact_marker.GetPrim())
    translate_op = xform.AddTranslateOp()
    
    for idx, frame_data in enumerate(frames):
        pos = frame_data["position"]
        time_code = float(idx)
        translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]), time_code)
    
    # Add metadata
    stage.GetRootLayer().customLayerData = {
        "contact_track_source": str(args.track),
        "num_frames": len(frames),
        "fps": args.fps,
    }
    
    stage.Save()
    print(f"[export_contact_track_to_usd] Wrote animated contact track to {args.output}")
    print(f"[export_contact_track_to_usd] {len(frames)} keyframes @ {args.fps} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
