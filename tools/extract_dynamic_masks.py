#!/usr/bin/env python3
"""Generate per-frame dynamic/static masks from a video using optical flow.

The goal is to segment moving elements (e.g. pantograph contact) versus
stationary background prior to COLMAP + Gaussian Splatting training.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the source video")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/temp_scene"),
        help="Base directory for extracted assets (default: data/temp_scene)",
    )
    parser.add_argument(
        "--mag-threshold",
        type=float,
        default=4.0,
        help="Flow magnitude threshold in pixels to classify motion (default: 4.0)",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=5,
        help="Morphological kernel size for mask cleanup (default: 5)",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="Process every Nth frame to trade accuracy for speed (default: 1)",
    )
    parser.add_argument(
        "--write-frames",
        action="store_true",
        help="Persist RGB frames alongside masks for inspection",
    )
    return parser.parse_args()


def _ensure_dirs(base: Path, write_frames: bool) -> tuple[Path, Path, Path | None]:
    dynamic_dir = base / "dynamic_masks"
    static_dir = base / "static_masks"
    frame_dir = base / "frames"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    if write_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)
        return dynamic_dir, static_dir, frame_dir
    return dynamic_dir, static_dir, None


def _flow_to_mask(flow: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = np.linalg.norm(flow, axis=2)
    dynamic_mask = (magnitude >= threshold).astype(np.uint8) * 255
    return dynamic_mask


def _clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def _write_mask(path: Path, array: np.ndarray) -> None:
    if not cv2.imwrite(str(path), array):
        raise RuntimeError(f"Failed to write mask to {path}")


def main() -> int:
    args = _parse_args()

    if not args.video.exists():
        print(f"[extract_dynamic_masks] Video not found: {args.video}", file=sys.stderr)
        return 1

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"[extract_dynamic_masks] Unable to open video: {args.video}", file=sys.stderr)
        return 1

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"[extract_dynamic_masks] Frames: {frame_count} @ {fps:.2f} fps")

    dynamic_dir, static_dir, frame_dir = _ensure_dirs(args.output, args.write_frames)

    success, prev_frame = capture.read()
    if not success:
        print("[extract_dynamic_masks] Failed to read first frame", file=sys.stderr)
        return 1

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_index = 0
    processed = 0

    iterator = tqdm(total=frame_count - 1, desc="Computing flow", unit="frame")

    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_index += 1
        iterator.update(1)

        if frame_index % args.sample_step:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        dynamic_mask = _flow_to_mask(flow, args.mag_threshold)
        dynamic_mask = _clean_mask(dynamic_mask, args.kernel)
        static_mask = cv2.bitwise_not(dynamic_mask)

        mask_name = f"frame_{frame_index:05d}.png"
        _write_mask(dynamic_dir / mask_name, dynamic_mask)
        _write_mask(static_dir / mask_name, static_mask)

        if frame_dir is not None:
            if not cv2.imwrite(str(frame_dir / f"frame_{frame_index:05d}.jpg"), frame):
                raise RuntimeError(f"Failed to write frame {frame_index}")

        prev_gray = gray
        processed += 1

    iterator.close()
    capture.release()

    if processed == 0:
        print("[extract_dynamic_masks] No frames processed (check sample-step)", file=sys.stderr)
        return 1

    print(
        "[extract_dynamic_masks] Masks written to"
        f" dynamic={dynamic_dir} static={static_dir} (processed {processed} frames)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
