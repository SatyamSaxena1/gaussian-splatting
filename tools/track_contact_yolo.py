#!/usr/bin/env python3
"""Track pantograph-catenary contact point using YOLO detection + depth maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Scene directory")
    parser.add_argument("model", type=Path, help="Trained YOLO model (.pt)")
    parser.add_argument("--output", type=Path, default=None, help="Output tracking JSON")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--visualize", action="store_true", help="Save visualization frames")
    return parser.parse_args()


def _load_depth(path: Path) -> np.ndarray:
    """Load normalized depth map."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        return None
    return depth.astype(np.float32) / np.iinfo(np.uint16).max


def _estimate_contact_3d(bbox: tuple[int, int, int, int], depth: np.ndarray, intrinsics: dict) -> tuple[float, float, float]:
    """
    Estimate 3D contact point from bounding box and depth.
    
    Args:
        bbox: (x1, y1, x2, y2) in pixels
        depth: Normalized depth map [0, 1]
        intrinsics: Camera parameters {fx, fy, cx, cy}
    
    Returns:
        (x, y, z) in camera coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Contact point is at top-center of pantograph bbox (where it touches wire)
    contact_u = (x1 + x2) // 2
    contact_v = y1  # Top edge
    
    # Sample depth in small region around contact point
    sample_v1 = max(0, contact_v - 2)
    sample_v2 = min(depth.shape[0], contact_v + 3)
    sample_u1 = max(0, contact_u - 2)
    sample_u2 = min(depth.shape[1], contact_u + 3)
    
    depth_sample = depth[sample_v1:sample_v2, sample_u1:sample_u2]
    if depth_sample.size == 0:
        return (0.0, 0.0, 0.0)
    
    # Use median depth for robustness
    d = np.median(depth_sample)
    if d == 0:
        return (0.0, 0.0, 0.0)
    
    # Back-project to 3D (assuming depth is inverse depth, closer = higher value)
    # Convert to actual depth (arbitrary scale)
    z = 1.0 / (d + 1e-6)
    
    fx = intrinsics.get("fx", 800.0)
    fy = intrinsics.get("fy", 800.0)
    cx = intrinsics.get("cx", depth.shape[1] / 2.0)
    cy = intrinsics.get("cy", depth.shape[0] / 2.0)
    
    x = (contact_u - cx) * z / fx
    y = (contact_v - cy) * z / fy
    
    return (float(x), float(y), float(z))


def main() -> int:
    args = _parse_args()
    
    frames_dir = args.scene / "frames"
    depth_dir = args.scene / "depth_maps"
    
    if not frames_dir.exists() or not depth_dir.exists():
        print("[track_contact_yolo] Missing frames or depth_maps directories", file=sys.stderr)
        return 1
    
    if not args.model.exists():
        print(f"[track_contact_yolo] Model not found: {args.model}", file=sys.stderr)
        return 1
    
    output_json = args.output or args.scene / "contact_track_yolo.json"
    vis_dir = args.scene / "contact_visualizations" if args.visualize else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model
    print(f"[track_contact_yolo] Loading model: {args.model}")
    model = YOLO(str(args.model))
    
    # Get frame list
    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        print(f"[track_contact_yolo] No frames found in {frames_dir}", file=sys.stderr)
        return 1
    
    print(f"[track_contact_yolo] Processing {len(frame_paths)} frames")
    print(f"[track_contact_yolo] Confidence threshold: {args.conf_threshold}")
    
    # Camera intrinsics (estimated - adjust based on video resolution)
    intrinsics = {
        "fx": 800.0,
        "fy": 800.0,
        "cx": 424.0,  # 848/2
        "cy": 239.0,  # 478/2
    }
    
    tracks = []
    
    for frame_path in tqdm(frame_paths, desc="Tracking", unit="frame"):
        name = frame_path.stem
        depth_path = depth_dir / f"{name}.png"
        
        if not depth_path.exists():
            tracks.append({
                "frame": name,
                "detected": False,
                "confidence": 0.0,
                "bbox": None,
                "contact_2d": None,
                "contact_3d": [0.0, 0.0, 0.0],
            })
            continue
        
        # Run YOLO detection
        results = model(str(frame_path), conf=args.conf_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            tracks.append({
                "frame": name,
                "detected": False,
                "confidence": 0.0,
                "bbox": None,
                "contact_2d": None,
                "contact_3d": [0.0, 0.0, 0.0],
            })
            continue
        
        # Get highest confidence detection
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confs)
        
        bbox_xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        confidence = float(confs[best_idx])
        
        x1, y1, x2, y2 = bbox_xyxy
        contact_u = (x1 + x2) // 2
        contact_v = y1
        
        # Load depth and estimate 3D position
        depth = _load_depth(depth_path)
        if depth is not None:
            contact_3d = _estimate_contact_3d(bbox_xyxy, depth, intrinsics)
        else:
            contact_3d = (0.0, 0.0, 0.0)
        
        tracks.append({
            "frame": name,
            "detected": True,
            "confidence": confidence,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "contact_2d": [int(contact_u), int(contact_v)],
            "contact_3d": list(contact_3d),
        })
        
        # Visualize if requested
        if vis_dir:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw contact point
                cv2.circle(frame, (contact_u, contact_v), 5, (0, 0, 255), -1)
                # Add text
                text = f"Conf: {confidence:.2f} | 3D: ({contact_3d[0]:.2f}, {contact_3d[1]:.2f}, {contact_3d[2]:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(str(vis_dir / f"{name}.jpg"), frame)
    
    # Compute statistics
    detected_count = sum(1 for t in tracks if t["detected"])
    avg_conf = np.mean([t["confidence"] for t in tracks if t["detected"]]) if detected_count > 0 else 0.0
    
    # Compute velocities (frame-to-frame)
    for i in range(1, len(tracks)):
        if tracks[i]["detected"] and tracks[i-1]["detected"]:
            p1 = np.array(tracks[i-1]["contact_3d"])
            p2 = np.array(tracks[i]["contact_3d"])
            velocity = p2 - p1
            tracks[i]["velocity_3d"] = velocity.tolist()
        else:
            tracks[i]["velocity_3d"] = [0.0, 0.0, 0.0]
    
    if len(tracks) > 0:
        tracks[0]["velocity_3d"] = [0.0, 0.0, 0.0]
    
    # Save results
    output_data = {
        "metadata": {
            "total_frames": len(tracks),
            "detected_frames": detected_count,
            "detection_rate": detected_count / len(tracks) if len(tracks) > 0 else 0.0,
            "avg_confidence": float(avg_conf),
            "model": str(args.model),
            "conf_threshold": args.conf_threshold,
        },
        "intrinsics": intrinsics,
        "tracks": tracks,
    }
    
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[track_contact_yolo] Results:")
    print(f"  Total frames: {len(tracks)}")
    print(f"  Detected: {detected_count} ({100*detected_count/len(tracks):.1f}%)")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"  Saved to: {output_json}")
    
    if vis_dir:
        print(f"  Visualizations: {vis_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
