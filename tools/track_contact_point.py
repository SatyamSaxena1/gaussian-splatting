#!/usr/bin/env python3
"""Track pantograph-catenary contact point using dynamic masks and static scene geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from plyfile import PlyData
from tqdm import tqdm


class ContactPoint(NamedTuple):
    frame: int
    position: tuple[float, float, float]
    confidence: float
    mask_area: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Scene directory (e.g. data/pantograph_scene)")
    parser.add_argument("--static-cloud", type=Path, required=True, help="Static scene point cloud PLY")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path for contact tracking data")
    parser.add_argument("--depth-threshold", type=float, default=0.3, help="Foreground depth percentile")
    parser.add_argument("--contact-zone-height", type=float, default=0.15, help="Fraction of frame height to consider for contact zone (top region)")
    return parser.parse_args()


def _load_static_cloud(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    positions = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T
    return positions.astype(np.float32)


def _load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise RuntimeError(f"Failed to load valid depth map: {path}")
    return depth.astype(np.float32) / np.iinfo(np.uint16).max


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to load mask: {path}")
    return mask


def _estimate_contact_position(
    dynamic_mask: np.ndarray,
    depth_map: np.ndarray,
    static_points: np.ndarray,
    contact_zone_height: float,
) -> tuple[tuple[float, float, float], float, int]:
    """Estimate 3D contact point from mask + depth in upper region of frame."""
    h, w = dynamic_mask.shape
    contact_zone_top = int(h * contact_zone_height)
    
    zone_mask = np.zeros_like(dynamic_mask)
    zone_mask[:contact_zone_top, :] = dynamic_mask[:contact_zone_top, :]
    
    contact_pixels = np.argwhere(zone_mask > 128)
    if len(contact_pixels) == 0:
        return (0.0, 0.0, 0.0), 0.0, 0
    
    centroid_y, centroid_x = contact_pixels.mean(axis=0)
    centroid_depth = depth_map[int(centroid_y), int(centroid_x)]
    
    # Simple heuristic: map 2D centroid + depth to approximate 3D position
    # In real pipeline, this would use camera intrinsics + extrinsics from COLMAP
    norm_x = (centroid_x / w - 0.5) * 2.0
    norm_y = (centroid_y / h - 0.5) * 2.0
    approx_z = centroid_depth * 10.0  # scale to scene units
    
    position = (float(norm_x), float(norm_y), float(approx_z))
    confidence = min(1.0, len(contact_pixels) / 1000.0)
    area = len(contact_pixels)
    
    return position, confidence, area


def main() -> int:
    args = _parse_args()
    
    if not args.static_cloud.exists():
        print(f"[track_contact_point] Static cloud not found: {args.static_cloud}", file=sys.stderr)
        return 1
    
    dynamic_mask_dir = args.scene / "dynamic_masks_refined"
    depth_dir = args.scene / "depth_maps"
    
    if not dynamic_mask_dir.exists() or not depth_dir.exists():
        print("[track_contact_point] Missing dynamic masks or depth maps", file=sys.stderr)
        return 1
    
    print("[track_contact_point] Loading static scene point cloud...")
    static_points = _load_static_cloud(args.static_cloud)
    print(f"[track_contact_point] Static cloud: {len(static_points)} points")
    
    mask_paths = sorted(dynamic_mask_dir.glob("*.png"))
    contact_track: list[ContactPoint] = []
    
    for mask_path in tqdm(mask_paths, desc="Tracking contact", unit="frame"):
        frame_name = mask_path.stem
        frame_num = int(frame_name.split("_")[-1])
        depth_path = depth_dir / f"{frame_name}.png"
        
        if not depth_path.exists():
            continue
        
        dynamic_mask = _load_mask(mask_path)
        depth_map = _load_depth(depth_path)
        
        position, confidence, area = _estimate_contact_position(
            dynamic_mask, depth_map, static_points, args.contact_zone_height
        )
        
        contact_track.append(ContactPoint(frame_num, position, confidence, area))
    
    output_path = args.output or args.scene / "contact_track.json"
    track_data = {
        "frames": [
            {
                "frame": cp.frame,
                "position": cp.position,
                "confidence": cp.confidence,
                "mask_area": cp.mask_area,
            }
            for cp in contact_track
        ],
        "metadata": {
            "static_cloud": str(args.static_cloud),
            "num_tracked_frames": len(contact_track),
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(track_data, f, indent=2)
    
    print(f"[track_contact_point] Tracked {len(contact_track)} frames → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
