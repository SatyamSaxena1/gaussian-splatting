import json
import argparse
import numpy as np
from pathlib import Path
import sys

def _parse_args():
    parser = argparse.ArgumentParser(description="Extract railway metrics from tracking data.")
    parser.add_argument("track_json", type=Path, help="Path to contact_track_yolo.json")
    parser.add_argument("--output", type=Path, help="Path to output metrics.json")
    parser.add_argument("--panto_width_mm", type=float, default=1600.0, help="Real-world width of pantograph head in mm")
    parser.add_argument("--camera_height_m", type=float, default=4.5, help="Height of camera above rail level in meters")
    parser.add_argument("--focal_length", type=float, default=800.0, help="Focal length in pixels (if not in JSON)")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    if not args.track_json.exists():
        print(f"Error: {args.track_json} not found", file=sys.stderr)
        return 1
        
    with open(args.track_json, 'r') as f:
        data = json.load(f)
        
    tracks = data.get("tracks", [])
    intrinsics = data.get("intrinsics", {})
    fx = intrinsics.get("fx", args.focal_length)
    fy = intrinsics.get("fy", args.focal_length)
    cx = intrinsics.get("cx", 0) # Should be image center
    cy = intrinsics.get("cy", 0)
    
    metrics = []
    
    print(f"Processing {len(tracks)} frames...")
    print(f"Parameters: Panto Width={args.panto_width_mm}mm, Camera Height={args.camera_height_m}m")
    
    prev_height = None
    
    for track in tracks:
        frame_metrics = {
            "frame": track["frame"],
            "detected": track["detected"],
            "height_m": None,
            "stagger_mm": None,
            "distance_m": None,
            "gradient_mm_per_frame": None
        }
        
        if track["detected"]:
            # 1. Scale Recovery using Bounding Box Width
            bbox = track["bbox"] # [x1, y1, x2, y2]
            w_px = bbox[2] - bbox[0]
            
            if w_px > 0:
                # Z = (f * W_real) / w_px
                # W_real is in mm, so Z will be in mm. Convert to meters.
                z_mm = (fx * args.panto_width_mm) / w_px
                z_m = z_mm / 1000.0
                
                # 2. 3D Position in Camera Frame (Metric)
                # Contact point (u, v)
                u, v = track["contact_2d"]
                
                # X = (u - cx) * Z / fx
                x_m = (u - cx) * z_m / fx
                
                # Y = (v - cy) * Z / fy
                # Note: Y is positive DOWN in image space.
                y_m = (v - cy) * z_m / fy
                
                # 3. Railway Metrics
                
                # Stagger: Lateral deviation (X)
                # Convert to mm
                stagger_mm = x_m * 1000.0
                
                # Height: Vertical distance from Rail Level
                # Camera is at +H_cam above rail.
                # Point is at Y_m relative to camera (Y down).
                # So Point Height = H_cam - Y_m
                # Example: Wire is above camera. v < cy -> y_m is negative.
                # Height = 4.5 - (-1.0) = 5.5m. Correct.
                height_m = args.camera_height_m - y_m
                
                # Implantation/Distance: Z distance from camera
                distance_m = z_m
                
                # Gradient: Change in height per frame
                gradient = 0.0
                if prev_height is not None:
                    gradient = (height_m - prev_height) * 1000.0 # mm per frame
                
                frame_metrics["height_m"] = round(height_m, 3)
                frame_metrics["stagger_mm"] = round(stagger_mm, 1)
                frame_metrics["distance_m"] = round(distance_m, 3)
                frame_metrics["gradient_mm_per_frame"] = round(gradient, 2)
                
                prev_height = height_m
                
        metrics.append(frame_metrics)
        
    # Save output
    output_path = args.output or args.track_json.with_name("railway_metrics.json")
    with open(output_path, 'w') as f:
        json.dump({"metrics": metrics, "parameters": vars(args)}, f, indent=2, default=str)
        
    print(f"Saved metrics to {output_path}")
    
    # Print some stats
    valid_heights = [m["height_m"] for m in metrics if m["height_m"] is not None]
    if valid_heights:
        print(f"\nStatistics:")
        print(f"  Avg Height: {np.mean(valid_heights):.3f} m")
        print(f"  Min Height: {np.min(valid_heights):.3f} m")
        print(f"  Max Height: {np.max(valid_heights):.3f} m")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
