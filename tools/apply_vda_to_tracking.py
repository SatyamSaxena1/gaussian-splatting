#!/usr/bin/env python3
"""
Apply Video Depth Anything depth maps to YOLO tracking detections.

This script replaces MiDaS depth with VDA depth in the tracking pipeline,
using metric depth values to calculate 3D positions.

Usage:
    python apply_vda_to_tracking.py --detections contact_detections.json --depth_dir vda_merged_depth/ --output contact_track_vda.json
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def load_depth_map(file_path: Path) -> np.ndarray:
    """Load depth map from various file formats."""
    ext = file_path.suffix.lower()
    
    if ext == '.npy':
        return np.load(file_path)
    elif ext == '.npz':
        data = np.load(file_path)
        for key in ['depth', 'depth_map', 'arr_0']:
            if key in data:
                return data[key]
        raise ValueError(f"Could not find depth data in NPZ file: {file_path}")
    elif ext in ['.png', '.jpg', '.tif', '.tiff']:
        depth = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Failed to load image: {file_path}")
        return depth
    else:
        raise ValueError(f"Unsupported depth map format: {ext}")


def find_depth_file(depth_dir: Path, frame_idx: int, format: str = "png16") -> Path:
    """Find depth file for a specific frame index."""
    # Try different naming patterns
    patterns = [
        f"depth_{frame_idx:06d}",  # depth_000000.png
        f"frame_{frame_idx:06d}",  # frame_000000.png
        f"{frame_idx:06d}",        # 000000.png
    ]
    
    extensions = {
        "png16": [".png"],
        "png8": [".png"],
        "npy": [".npy"],
        "npz": [".npz"]
    }
    
    exts = extensions.get(format, [".png"])
    
    for pattern in patterns:
        for ext in exts:
            candidate = depth_dir / f"{pattern}{ext}"
            if candidate.exists():
                return candidate
    
    raise FileNotFoundError(f"Depth file not found for frame {frame_idx} in {depth_dir}")


def denormalize_depth(depth: np.ndarray, format: str = "png16") -> np.ndarray:
    """
    Convert normalized depth back to metric values if needed.
    
    Video Depth Anything produces metric depth, but PNG saves may normalize it.
    """
    if format in ["png16", "png8"]:
        # If depth was normalized to 0-65535 or 0-255, we need metadata to denormalize
        # For now, assume depth is already in meters or normalized to reasonable range
        # This may need adjustment based on actual VDA output format
        if depth.max() > 100:
            # Likely normalized to uint16 range, need to scale back
            # This is a heuristic - may need adjustment
            depth = depth.astype(np.float32) / 65535.0 * 10.0  # Assume 0-10m range
    
    return depth.astype(np.float32)


def get_depth_at_bbox(depth: np.ndarray, bbox: List[float]) -> Tuple[float, float]:
    """
    Extract depth value at bounding box location.
    
    Args:
        depth: Depth map array (H, W)
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
    
    Returns:
        Tuple of (median_depth, std_depth)
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure bbox is within image bounds
    h, w = depth.shape
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    
    # Extract depth patch
    depth_patch = depth[y1:y2+1, x1:x2+1]
    
    if depth_patch.size == 0:
        return 0.0, 0.0
    
    # Use median for robustness against outliers
    median_depth = float(np.median(depth_patch))
    std_depth = float(np.std(depth_patch))
    
    return median_depth, std_depth


def pixel_to_3d(bbox: List[float], depth: float, fx: float, fy: float, 
                cx: float, cy: float) -> Tuple[float, float, float]:
    """
    Convert bounding box center and depth to 3D position.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        depth: Depth value in meters
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point in pixels
    
    Returns:
        3D position (x, y, z) in meters
    """
    # Get bbox center
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    
    # Convert to 3D using pinhole camera model
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return x, y, z


def apply_vda_depth(detections: Dict, depth_dir: Path, format: str = "png16",
                   camera_params: Dict = None) -> Dict:
    """
    Apply VDA depth maps to tracking detections.
    
    Args:
        detections: Detection dictionary from YOLO tracking
        depth_dir: Directory containing VDA depth maps
        format: Depth map format (png16/png8/npy/npz)
        camera_params: Camera intrinsic parameters (optional)
    
    Returns:
        Updated detection dictionary with 3D positions
    """
    # Default camera parameters (estimated from video)
    if camera_params is None:
        # These are rough estimates - should be calibrated for your camera
        img_width = detections.get("image_width", 1920)
        img_height = detections.get("image_height", 1080)
        
        # Typical smartphone camera FOV ~70 degrees horizontal
        fov_h = 70 * np.pi / 180
        fx = img_width / (2 * np.tan(fov_h / 2))
        fy = fx  # Assume square pixels
        cx = img_width / 2
        cy = img_height / 2
    else:
        fx = camera_params["fx"]
        fy = camera_params["fy"]
        cx = camera_params["cx"]
        cy = camera_params["cy"]
    
    print(f"Camera parameters:")
    print(f"  fx={fx:.1f}, fy={fy:.1f}")
    print(f"  cx={cx:.1f}, cy={cy:.1f}")
    
    # Create output structure
    output = {
        "metadata": detections.get("metadata", {}),
        "image_width": detections.get("image_width"),
        "image_height": detections.get("image_height"),
        "camera_params": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "depth_source": "Video-Depth-Anything",
        "frames": []
    }
    
    frames = detections.get("frames", [])
    
    print(f"\nProcessing {len(frames)} frames...")
    
    for frame_data in tqdm(frames):
        frame_idx = frame_data["frame_index"]
        
        # Load corresponding depth map
        try:
            depth_file = find_depth_file(depth_dir, frame_idx, format)
            depth_map = load_depth_map(depth_file)
            depth_map = denormalize_depth(depth_map, format)
        except FileNotFoundError as e:
            print(f"\nWarning: {e}")
            # Skip this frame or use placeholder
            continue
        except Exception as e:
            print(f"\nError loading depth for frame {frame_idx}: {e}")
            continue
        
        # Process detections in this frame
        frame_output = {
            "frame_index": frame_idx,
            "timestamp": frame_data.get("timestamp"),
            "detections": []
        }
        
        for detection in frame_data.get("detections", []):
            bbox = detection["bbox"]
            
            # Get depth at bbox
            depth, depth_std = get_depth_at_bbox(depth_map, bbox)
            
            # Convert to 3D
            x, y, z = pixel_to_3d(bbox, depth, fx, fy, cx, cy)
            
            # Add 3D information to detection
            detection_output = {
                **detection,  # Copy all existing fields
                "depth": depth,
                "depth_std": depth_std,
                "position_3d": {"x": x, "y": y, "z": z}
            }
            
            frame_output["detections"].append(detection_output)
        
        output["frames"].append(frame_output)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Apply VDA depth to YOLO tracking")
    parser.add_argument("--detections", required=True, help="Path to detection JSON file")
    parser.add_argument("--depth_dir", required=True, help="Directory with VDA depth maps")
    parser.add_argument("--output", required=True, help="Output JSON file with 3D positions")
    parser.add_argument("--format", choices=["png16", "png8", "npy", "npz"], default="png16",
                       help="Depth map format")
    parser.add_argument("--fx", type=float, help="Focal length x (optional)")
    parser.add_argument("--fy", type=float, help="Focal length y (optional)")
    parser.add_argument("--cx", type=float, help="Principal point x (optional)")
    parser.add_argument("--cy", type=float, help="Principal point y (optional)")
    
    args = parser.parse_args()
    
    # Load detections
    print(f"Loading detections from: {args.detections}")
    with open(args.detections, 'r') as f:
        detections = json.load(f)
    
    # Camera parameters
    camera_params = None
    if all([args.fx, args.fy, args.cx, args.cy]):
        camera_params = {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy
        }
        print("Using provided camera parameters")
    else:
        print("Using estimated camera parameters (consider calibrating for better accuracy)")
    
    # Apply VDA depth
    print(f"Loading depth maps from: {args.depth_dir}")
    output = apply_vda_depth(
        detections=detections,
        depth_dir=Path(args.depth_dir),
        format=args.format,
        camera_params=camera_params
    )
    
    # Save output
    print(f"\nSaving 3D tracking data to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print statistics
    total_detections = sum(len(f["detections"]) for f in output["frames"])
    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"{'='*60}")
    print(f"Frames processed: {len(output['frames'])}")
    print(f"Total detections: {total_detections}")
    
    if total_detections > 0:
        # Calculate depth statistics
        all_depths = []
        all_z = []
        for frame in output["frames"]:
            for det in frame["detections"]:
                all_depths.append(det["depth"])
                all_z.append(det["position_3d"]["z"])
        
        print(f"\nDepth statistics:")
        print(f"  Min: {np.min(all_depths):.3f}m")
        print(f"  Max: {np.max(all_depths):.3f}m")
        print(f"  Mean: {np.mean(all_depths):.3f}m")
        print(f"  Median: {np.median(all_depths):.3f}m")
        
        print(f"\nZ-coordinate range: {np.min(all_z):.3f}m to {np.max(all_z):.3f}m")
        print(f"Z-axis span: {np.max(all_z) - np.min(all_z):.3f}m")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
