#!/usr/bin/env python3
"""Generate per-frame depth maps using MiDaS on extracted video frames.

Expected input directory layout (from extract_dynamic_masks step):

    data/<scene_name>/frames/frame_00001.jpg

Depth outputs are written alongside frames:

    data/<scene_name>/depth_maps/frame_00001.png (16-bit depth)

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "frames",
        type=Path,
        help="Directory containing RGB frames (e.g. data/pantograph_scene/frames)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory for depth maps (defaults to <frames>/../depth_maps)",
    )
    parser.add_argument(
        "--model",
        default="DPT_BEiT_L_384",
        choices=["DPT_BEiT_L_384", "DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model variant (default: DPT_BEiT_L_384)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional max size for the longer frame edge to speed up inference",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Persist npy files with the raw floating-point depth values",
    )
    return parser.parse_args()


def _load_model(model_name: str) -> tuple[torch.nn.Module, torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", model_name)
    model.to(device)
    model.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name in ("DPT_BEiT_L_384", "DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return model, transform, device


def _resize_frame(frame: np.ndarray, max_edge: int | None) -> np.ndarray:
    if max_edge is None:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return frame
    scale = max_edge / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    if depth_max - depth_min <= 1e-6:
        return np.zeros_like(depth, dtype=np.uint16)
    normalized = (depth - depth_min) / (depth_max - depth_min)
    scaled = (normalized * np.iinfo(np.uint16).max).astype(np.uint16)
    return scaled


def main() -> int:
    args = _parse_args()

    frame_dir = args.frames
    if not frame_dir.exists():
        print(f"[generate_depth_maps] Frame directory not found: {frame_dir}", file=sys.stderr)
        return 1

    image_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    if not image_paths:
        print(f"[generate_depth_maps] No frames discovered in {frame_dir}", file=sys.stderr)
        return 1

    output_dir = args.output or frame_dir.parent / "depth_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir: Path | None = None
    if args.save_raw:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate_depth_maps] Loading MiDaS model ({args.model})")
    model, transform, device = _load_model(args.model)
    print(f"[generate_depth_maps] Using device: {device}")

    for frame_path in tqdm(image_paths, desc="Depth inference", unit="frame"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[generate_depth_maps] Failed to read {frame_path}", file=sys.stderr)
            continue

        frame_resized = _resize_frame(frame, args.resize)
        input_batch = transform(frame_resized).to(device)

        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_resized.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        if frame_resized.shape[:2] != frame.shape[:2]:
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        if raw_dir is not None:
            np.save(raw_dir / (frame_path.stem + ".npy"), depth_map)

        depth_u16 = _normalize_depth(depth_map)
        depth_path = output_dir / (frame_path.stem + ".png")
        if not cv2.imwrite(str(depth_path), depth_u16):
            print(f"[generate_depth_maps] Failed to write {depth_path}", file=sys.stderr)

    print(f"[generate_depth_maps] Depth maps saved to {output_dir}")
    if raw_dir is not None:
        print(f"[generate_depth_maps] Raw depth arrays saved to {raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
