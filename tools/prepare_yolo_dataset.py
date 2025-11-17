#!/usr/bin/env python3
"""Generate YOLO dataset from foreground masks for pantograph detection."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Scene directory")
    parser.add_argument("--output", type=Path, default=None, help="Output YOLO dataset directory")
    parser.add_argument("--num-train", type=int, default=180, help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--sample-step", type=int, default=None, help="Sample every Nth frame (auto if None)")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum mask area to generate label")
    return parser.parse_args()


def _mask_to_yolo_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Convert binary mask to YOLO bbox format (x_center, y_center, width, height) normalized."""
    # Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    
    # Get largest component (skip background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    
    if areas[largest_label - 1] < 500:  # Minimum area threshold
        return None
    
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    
    img_h, img_w = mask.shape
    
    # Convert to YOLO format (normalized center x, center y, width, height)
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    width = w / img_w
    height = h / img_h
    
    return (x_center, y_center, width, height)


def _write_yolo_label(path: Path, bbox: tuple[float, float, float, float]) -> None:
    """Write YOLO label file (class_id x_center y_center width height)."""
    with open(path, "w") as f:
        f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def main() -> int:
    args = _parse_args()
    
    frames_dir = args.scene / "frames"
    masks_dir = args.scene / "foreground_masks"
    
    if not frames_dir.exists():
        print("[prepare_yolo_dataset] Missing frames directory", file=sys.stderr)
        return 1
    
    use_masks = masks_dir.exists()
    
    output_dir = args.output or Path("data/pantograph_yolo")
    
    # Create YOLO directory structure
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all frame paths
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    total_samples = args.num_train + args.num_val
    
    if len(frame_paths) < total_samples:
        print(f"[prepare_yolo_dataset] Only {len(frame_paths)} frames available, need {total_samples}", file=sys.stderr)
        return 1
    
    # Sample frames
    if args.sample_step:
        sampled = frame_paths[::args.sample_step][:total_samples]
    else:
        step = len(frame_paths) // total_samples
        sampled = frame_paths[::step][:total_samples]
    
    random.shuffle(sampled)
    train_frames = sampled[:args.num_train]
    val_frames = sampled[args.num_train:]
    
    print(f"[prepare_yolo_dataset] Generating {args.num_train} train + {args.num_val} val samples")
    
    for split, frame_list in [("train", train_frames), ("val", val_frames)]:
        skipped = 0
        for frame_path in tqdm(frame_list, desc=f"Processing {split}", unit="frame"):
            name = frame_path.stem
            
            # Copy image (always)
            dst_image = output_dir / "images" / split / f"{name}.jpg"
            shutil.copy(frame_path, dst_image)
            
            # Try to auto-generate label from mask if available
            if use_masks:
                mask_path = masks_dir / f"{name}.png"
                
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        bbox = _mask_to_yolo_bbox(mask)
                        if bbox is not None:
                            dst_label = output_dir / "labels" / split / f"{name}.txt"
                            _write_yolo_label(dst_label, bbox)
                            continue
            
            # If no mask or auto-label failed, copy frame without label (for manual labeling)
            skipped += 1
        
        if skipped > 0:
            print(f"[prepare_yolo_dataset] {skipped} {split} frames need manual labeling")
    
    # Write YOLO config file
    config_path = output_dir / "pantograph.yaml"
    config_content = f"""# Pantograph Detection Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: pantograph

# Augmentation (optional, YOLO applies defaults)
# hsv_h: 0.015
# hsv_s: 0.7
# hsv_v: 0.4
# degrees: 0.0
# translate: 0.1
# scale: 0.5
# flipud: 0.0
# fliplr: 0.5
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"[prepare_yolo_dataset] Dataset created at {output_dir}")
    print(f"[prepare_yolo_dataset] Config file: {config_path}")
    print(f"\n[prepare_yolo_dataset] Next steps:")
    print(f"  1. Review labels: labelImg {output_dir / 'images' / 'train'}")
    print(f"  2. Train YOLO: yolo detect train model=yolov8n.pt data={config_path} epochs=100 imgsz=640")
    return 0


if __name__ == "__main__":
    sys.exit(main())
