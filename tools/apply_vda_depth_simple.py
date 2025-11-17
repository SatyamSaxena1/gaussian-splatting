#!/usr/bin/env python3
"""
Simple script to apply VDA depth to pantograph tracking data.
Adapts the existing tracking JSON format to include VDA depth.
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def load_depth_map(file_path: Path) -> np.ndarray:
    """Load depth map from PNG16."""
    depth = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise ValueError(f"Failed to load: {file_path}")
    # Convert from uint16 (0-65535) back to relative depth
    return depth.astype(np.float32) / 65535.0


def find_depth_file(depth_dir: Path, frame_name: str) -> Path:
    """Find depth file for a frame name like 'frame_00001'."""
    # Extract frame number
    frame_num = int(frame_name.split('_')[-1])
    
    # Try depth_XXXXXX.png format
    depth_file = depth_dir / f"depth_{frame_num:06d}.png"
    if depth_file.exists():
        return depth_file
    
    raise FileNotFoundError(f"Depth file not found for {frame_name}")


def get_depth_at_point(depth_map: np.ndarray, x: int, y: int, window=5) -> tuple:
    """Get depth at a point with small window for robustness."""
    h, w = depth_map.shape
    
    # Clamp coordinates
    x = max(0, min(w-1, x))
    y = max(0, min(h-1, y))
    
    # Get window around point
    y1 = max(0, y - window//2)
    y2 = min(h, y + window//2 + 1)
    x1 = max(0, x - window//2)
    x2 = min(w, x + window//2 + 1)
    
    window_depths = depth_map[y1:y2, x1:x2]
    
    # Remove zeros and get median
    valid_depths = window_depths[window_depths > 0]
    if len(valid_depths) == 0:
        return 0.0, 0.0
    
    depth = np.median(valid_depths)
    depth_std = np.std(valid_depths)
    
    return float(depth), float(depth_std)


def pixel_to_3d(x, y, depth, fx, fy, cx, cy):
    """Convert pixel coordinates + depth to 3D position."""
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return X, Y, Z


def main():
    parser = argparse.ArgumentParser(description="Apply VDA depth to pantograph tracking")
    parser.add_argument("--tracking", required=True, help="Input tracking JSON")
    parser.add_argument("--depth_dir", required=True, help="Directory with VDA depth PNGs")
    parser.add_argument("--output", required=True, help="Output JSON with VDA depth")
    
    args = parser.parse_args()
    
    depth_dir = Path(args.depth_dir)
    
    # Load tracking data
    print(f"Loading tracking data from: {args.tracking}")
    with open(args.tracking, 'r') as f:
        tracking = json.load(f)
    
    # Get camera intrinsics
    intrinsics = tracking.get("intrinsics", {})
    fx = intrinsics.get("fx", 800.0)
    fy = intrinsics.get("fy", 800.0)
    cx = intrinsics.get("cx", 424.0)
    cy = intrinsics.get("cy", 239.0)
    
    print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Process tracks
    tracks = tracking.get("tracks", [])
    print(f"Processing {len(tracks)} tracks...")
    
    processed = 0
    skipped = 0
    
    for track in tqdm(tracks):
        if not track.get("detected", False):
            skipped += 1
            continue
        
        frame_name = track["frame"]
        contact_2d = track.get("contact_2d")
        
        if contact_2d is None:
            skipped += 1
            continue
        
        try:
            # Load depth map
            depth_file = find_depth_file(depth_dir, frame_name)
            depth_map = load_depth_map(depth_file)
            
            # Get depth at contact point
            x, y = contact_2d
            depth_val, depth_std = get_depth_at_point(depth_map, x, y)
            
            # Convert to 3D
            X, Y, Z = pixel_to_3d(x, y, depth_val, fx, fy, cx, cy)
            
            # Add VDA depth information
            track["vda_depth"] = depth_val
            track["vda_depth_std"] = depth_std
            track["vda_position_3d"] = {
                "x": X,
                "y": Y,
                "z": Z
            }
            
            processed += 1
            
        except FileNotFoundError:
            skipped += 1
            continue
        except Exception as e:
            print(f"\nError processing {frame_name}: {e}")
            skipped += 1
            continue
    
    # Update metadata
    tracking["metadata"]["vda_processed"] = processed
    tracking["metadata"]["vda_skipped"] = skipped
    tracking["metadata"]["depth_source"] = "Video-Depth-Anything-v1.3.1"
    
    # Save output
    print(f"\nSaving to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(tracking, f, indent=2)
    
    print("\n" + "="*60)
    print("VDA Depth Application Complete!")
    print("="*60)
    print(f"Processed: {processed} tracks")
    print(f"Skipped: {skipped} tracks")
    print(f"Success rate: {processed/(processed+skipped)*100:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
