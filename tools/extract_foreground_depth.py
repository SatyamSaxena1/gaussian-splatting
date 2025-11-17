#!/usr/bin/env python3
"""Extract foreground using depth-only thresholding for pantograph segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Scene directory (e.g. data/pantograph_scene)")
    parser.add_argument(
        "--depth-percentile",
        type=float,
        default=25.0,
        help="Depth percentile threshold for foreground (lower = closer, default: 25.0)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum connected component area to keep (default: 500)",
    )
    parser.add_argument(
        "--morphology-kernel",
        type=int,
        default=5,
        help="Kernel size for morphological operations (default: 5)",
    )
    return parser.parse_args()


def _load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise RuntimeError(f"Failed to load depth map: {path}")
    return depth.astype(np.float32) / np.iinfo(np.uint16).max


def _load_frame(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Failed to load frame: {path}")
    return frame


def _compute_foreground_mask(
    depth: np.ndarray,
    percentile: float,
    min_area: int,
    kernel_size: int,
) -> np.ndarray:
    """Compute foreground mask from depth using percentile threshold."""
    flat = depth.flatten()
    threshold = np.percentile(flat, percentile)
    
    # Pixels closer than threshold are foreground
    mask = (depth <= threshold).astype(np.uint8) * 255
    
    # Morphological operations to clean up mask
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small components
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == label_id] = 255
        mask = filtered
    
    return mask


def main() -> int:
    args = _parse_args()
    
    frames_dir = args.scene / "frames"
    depth_dir = args.scene / "depth_maps"
    
    if not frames_dir.exists() or not depth_dir.exists():
        print("[extract_foreground_depth] Missing frames or depth_maps directories", file=sys.stderr)
        return 1
    
    output_masks = args.scene / "foreground_masks"
    output_frames = args.scene / "foreground_frames"
    output_masks.mkdir(parents=True, exist_ok=True)
    output_frames.mkdir(parents=True, exist_ok=True)
    
    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        print(f"[extract_foreground_depth] No frames found in {frames_dir}", file=sys.stderr)
        return 1
    
    print(f"[extract_foreground_depth] Processing {len(frame_paths)} frames")
    print(f"[extract_foreground_depth] Depth percentile: {args.depth_percentile}%")
    
    for frame_path in tqdm(frame_paths, desc="Extracting foreground", unit="frame"):
        name = frame_path.stem
        depth_path = depth_dir / f"{name}.png"
        
        if not depth_path.exists():
            continue
        
        frame = _load_frame(frame_path)
        depth = _load_depth(depth_path)
        
        mask = _compute_foreground_mask(
            depth,
            args.depth_percentile,
            args.min_area,
            args.morphology_kernel,
        )
        
        # Save mask
        cv2.imwrite(str(output_masks / f"{name}.png"), mask)
        
        # Apply mask to frame
        foreground = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite(str(output_frames / f"{name}.jpg"), foreground)
    
    print(f"[extract_foreground_depth] Foreground masks saved to {output_masks}")
    print(f"[extract_foreground_depth] Foreground frames saved to {output_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
