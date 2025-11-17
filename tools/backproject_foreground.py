#!/usr/bin/env python3
"""Back-project foreground depth masks to 3D point cloud using COLMAP camera parameters."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Scene directory")
    parser.add_argument("--output", type=Path, default=None, help="Output PLY path")
    parser.add_argument("--sample-step", type=int, default=10, help="Sample every Nth frame to reduce point count")
    return parser.parse_args()


def _read_colmap_cameras(cameras_bin: Path) -> dict:
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(cameras_bin, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            params = struct.unpack("<" + "d" * 4, f.read(8 * 4))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,  # fx, fy, cx, cy for OPENCV model
            }
    return cameras


def _read_colmap_images(images_bin: Path) -> dict:
    """Read COLMAP images.bin file."""
    images = {}
    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_len = 0
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")
            num_points = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points)  # skip point data
            images[name] = {
                "image_id": image_id,
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
            }
    return images


def _qvec_to_rotation(qvec):
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def _backproject_frame(
    mask: np.ndarray,
    depth: np.ndarray,
    frame: np.ndarray,
    camera_params: tuple,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project masked depth to 3D points."""
    fx, fy, cx, cy = camera_params
    h, w = mask.shape
    
    # Get foreground pixels
    ys, xs = np.where(mask > 128)
    if len(xs) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # Get depth and color
    depths = depth[ys, xs]
    colors = frame[ys, xs] / 255.0
    
    # Unproject to camera coordinates
    x_cam = (xs - cx) * depths / fx
    y_cam = (ys - cy) * depths / fy
    z_cam = depths
    
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Transform to world coordinates
    points_world = (R.T @ (points_cam.T - t[:, None])).T
    
    return points_world.astype(np.float32), colors.astype(np.float32)


def main() -> int:
    args = _parse_args()
    
    sparse_dir = args.scene / "sparse" / "0"
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    
    if not cameras_bin.exists() or not images_bin.exists():
        print("[backproject_foreground] COLMAP sparse reconstruction not found", file=sys.stderr)
        return 1
    
    print("[backproject_foreground] Loading COLMAP cameras and images...")
    cameras = _read_colmap_cameras(cameras_bin)
    images = _read_colmap_images(images_bin)
    
    mask_dir = args.scene / "foreground_masks"
    depth_dir = args.scene / "depth_maps"
    frame_dir = args.scene / "frames"
    
    mask_paths = sorted(mask_dir.glob("*.png"))[::args.sample_step]
    
    all_points = []
    all_colors = []
    
    print(f"[backproject_foreground] Back-projecting {len(mask_paths)} frames...")
    for mask_path in tqdm(mask_paths, desc="Back-projecting", unit="frame"):
        name = mask_path.stem
        image_name = f"{name}.jpg"
        
        if image_name not in images:
            continue
        
        depth_path = depth_dir / f"{name}.png"
        frame_path = frame_dir / f"{name}.jpg"
        
        if not depth_path.exists() or not frame_path.exists():
            continue
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        frame = cv2.imread(str(frame_path))
        
        image_data = images[image_name]
        camera = cameras[image_data["camera_id"]]
        
        R = _qvec_to_rotation(image_data["qvec"])
        t = np.array(image_data["tvec"])
        
        points, colors = _backproject_frame(mask, depth, frame, camera["params"], R, t)
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
    
    if not all_points:
        print("[backproject_foreground] No points generated", file=sys.stderr)
        return 1
    
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    
    print(f"[backproject_foreground] Generated {len(all_points)} points")
    
    # Write PLY
    output_path = args.output or args.scene / "foreground_cloud.ply"
    vertex = np.array(
        [(p[0], p[1], p[2], int(c[2]*255), int(c[1]*255), int(c[0]*255))
         for p, c in zip(all_points, all_colors)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(str(output_path))
    
    print(f"[backproject_foreground] Wrote point cloud to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
