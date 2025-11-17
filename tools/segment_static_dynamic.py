#!/usr/bin/env python3
"""Split extracted frames into static and dynamic sets using motion + depth cues."""

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
    parser.add_argument("--dynamic-threshold", type=float, default=0.5, help="Minimum mask value to treat pixel as dynamic (0-1 range)")
    parser.add_argument("--depth-threshold", type=float, default=0.35, help="Foreground percentile for depth separation (0-1)")
    parser.add_argument("--depth-mode", choices=["percentile", "absolute"], default="percentile", help="Interpret depth threshold as percentile or absolute normalized value")
    parser.add_argument("--min-area", type=int, default=500, help="Ignore dynamic regions smaller than this many pixels")
    parser.add_argument("--erode", type=int, default=3, help="Morphological erosion kernel size for foreground mask cleanup")
    parser.add_argument("--output-static", type=Path, default=None, help="Override output directory for static frames")
    parser.add_argument("--output-dynamic", type=Path, default=None, help="Override output directory for dynamic frames")
    return parser.parse_args()


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return image


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to load mask: {path}")
    return mask.astype(np.float32) / 255.0


def _load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to load depth map: {path}")
    if depth.dtype != np.uint16:
        raise RuntimeError(f"Expected 16-bit depth map, got {depth.dtype} at {path}")
    return depth.astype(np.float32) / np.iinfo(np.uint16).max


def _compute_depth_cutoff(depth: np.ndarray, threshold: float, mode: str) -> float:
    flat = depth.flatten()
    if mode == "percentile":
        return np.percentile(flat, threshold * 100.0)
    return threshold


def _filter_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label_id] = 255
    return filtered


def _postprocess_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _ensure_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = _parse_args()

    frame_dir = args.scene / "frames"
    dynamic_dir = args.scene / "dynamic_masks"
    depth_dir = args.scene / "depth_maps"

    if not frame_dir.exists() or not dynamic_dir.exists() or not depth_dir.exists():
        print("[segment_static_dynamic] Missing required inputs (frames, dynamic masks, or depth maps)", file=sys.stderr)
        return 1

    frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        print(f"[segment_static_dynamic] No frames found in {frame_dir}", file=sys.stderr)
        return 1

    static_out = args.output_static or args.scene / "static_frames"
    dynamic_out = args.output_dynamic or args.scene / "dynamic_frames"
    static_mask_out = args.scene / "static_masks_refined"
    dynamic_mask_out = args.scene / "dynamic_masks_refined"

    for directory in (static_out, dynamic_out, static_mask_out, dynamic_mask_out):
        _ensure_output(directory)

    for frame_path in tqdm(frame_paths, desc="Segmenting", unit="frame"):
        name = frame_path.stem
        dynamic_mask_path = dynamic_dir / f"{name}.png"
        depth_path = depth_dir / f"{name}.png"

        if not dynamic_mask_path.exists() or not depth_path.exists():
            continue

        frame = _load_image(frame_path)
        dyn_mask = _load_mask(dynamic_mask_path)
        depth = _load_depth(depth_path)

        depth_cutoff = _compute_depth_cutoff(depth, args.depth_threshold, args.depth_mode)
        depth_foreground = (depth <= depth_cutoff).astype(np.float32)

        dynamic_binary = (dyn_mask >= args.dynamic_threshold).astype(np.float32)
        foreground = dynamic_binary * depth_foreground

        foreground = _filter_small_regions(foreground, args.min_area)
        foreground = _postprocess_mask(foreground, args.erode)
        background = cv2.bitwise_not(foreground)

        cv2.imwrite(str(dynamic_mask_out / f"{name}.png"), foreground)
        cv2.imwrite(str(static_mask_out / f"{name}.png"), background)

        dynamic_frame = cv2.bitwise_and(frame, frame, mask=foreground)
        static_frame = cv2.bitwise_and(frame, frame, mask=background)

        cv2.imwrite(str(dynamic_out / f"{name}.jpg"), dynamic_frame)
        cv2.imwrite(str(static_out / f"{name}.jpg"), static_frame)

    print(f"[segment_static_dynamic] Static frames written to {static_out}")
    print(f"[segment_static_dynamic] Dynamic frames written to {dynamic_out}")
    print(f"[segment_static_dynamic] Refined masks stored under {static_mask_out} and {dynamic_mask_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
