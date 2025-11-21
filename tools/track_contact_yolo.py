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
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to process")
    return parser.parse_args()


def _load_depth(path: Path) -> np.ndarray:
    """Load normalized depth map."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        return None
    return depth.astype(np.float32) / np.iinfo(np.uint16).max


def _estimate_contact_3d(bbox: tuple[int, int, int, int], depth: np.ndarray, intrinsics: dict) -> tuple[float, float, float, float, float]:
    """
    Estimate 3D contact point from bounding box and depth.
    
    Args:
        bbox: (x1, y1, x2, y2) in pixels
        depth: Normalized depth map [0, 1]
        intrinsics: Camera parameters {fx, fy, cx, cy}
    
    Returns:
        (x, y, z, depth_val, depth_std) in camera coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # Contact point is at top-center of pantograph bbox (where it touches wire)
    contact_u = (x1 + x2) // 2
    contact_v = y1  # Top edge
    
    # Sample depth in small region around contact point
    # OFFSET: Shift window slightly DOWN (y+5) to sample pantograph body, not sky
    sample_center_v = min(depth.shape[0]-1, contact_v + 5)
    
    sample_v1 = max(0, sample_center_v - 2)
    sample_v2 = min(depth.shape[0], sample_center_v + 3)
    sample_u1 = max(0, contact_u - 2)
    sample_u2 = min(depth.shape[1], contact_u + 3)
    
    depth_sample = depth[sample_v1:sample_v2, sample_u1:sample_u2]
    if depth_sample.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Robust sampling: Use 10th percentile to find "closest" object (pantograph)
    # This avoids averaging with the background sky (which has high depth value / low inverse depth)
    # Note: Depth map is likely inverse depth or disparity? 
    # VDA usually outputs relative depth. If it's inverse depth (closer = higher), we want MAX.
    # If it's metric depth (closer = lower), we want MIN.
    # PIPELINE_DOC says: "Inverse depth representation (closer = higher value)"
    # So we want the HIGHER values (closer objects).
    
    valid_depths = depth_sample[depth_sample > 0]
    if valid_depths.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
        
    # Use 90th percentile (closer objects have higher values in inverse depth)
    d = np.percentile(valid_depths, 90)
    d_std = np.std(valid_depths)
    
    if d == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Back-project to 3D
    # z = 1.0 / d  (assuming d is inverse depth)
    z = 1.0 / (d + 1e-6)
    
    fx = intrinsics.get("fx", 800.0)
    fy = intrinsics.get("fy", 800.0)
    cx = intrinsics.get("cx", depth.shape[1] / 2.0)
    cy = intrinsics.get("cy", depth.shape[0] / 2.0)
    
    x = (contact_u - cx) * z / fx
    y = (contact_v - cy) * z / fy
    
    return (float(x), float(y), float(z), float(d), float(d_std))


def main() -> int:
    args = _parse_args()
    
    # Try to find frames directory
    frames_dir = args.scene / "frames"
    if not frames_dir.exists():
        frames_dir = args.scene / "input"
    if not frames_dir.exists():
        frames_dir = args.scene / "images"
        
    depth_dir = args.scene / "depth_maps"
    if not depth_dir.exists():
        depth_dir = args.scene / "vda_merged_depth"
    
    if not frames_dir.exists():
        print(f"[track_contact_yolo] Missing frames/input/images directory in {args.scene}", file=sys.stderr)
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
    
    if args.limit:
        frame_paths = frame_paths[:args.limit]
    
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
    
    # Optical Flow State
    prev_gray = None
    prev_pts = None
    prev_bbox = None
    
    # LK Params
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    for frame_path in tqdm(frame_paths, desc="Tracking", unit="frame"):
        name = frame_path.stem
        
        # Load frame
        frame_img = cv2.imread(str(frame_path))
        if frame_img is None:
            continue
        frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        
        # Try different depth map naming conventions
        depth_path = depth_dir / f"{name}.png"
        if not depth_path.exists():
            if name.startswith("frame_"):
                suffix = name.replace("frame_", "")
                depth_path = depth_dir / f"depth_{suffix}.png"
            else:
                depth_path = depth_dir / f"depth_{name}.png"
        
        # 1. Run YOLO detection
        results = model(str(frame_path), conf=args.conf_threshold, verbose=False)
        
        detected = False
        bbox = None
        confidence = 0.0
        
        # Check YOLO results
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            cls_ids = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            # Find best pantograph (class 0)
            best_conf = -1.0
            best_idx = -1
            
            for i, (cls_id, conf) in enumerate(zip(cls_ids, confs)):
                if cls_id == 0 and conf > best_conf:
                    best_conf = conf
                    best_idx = i
            
            if best_idx != -1:
                detected = True
                confidence = float(best_conf)
                bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        
        # 2. Optical Flow Fallback
        is_optical_flow = False
        if not detected and prev_pts is not None and prev_gray is not None:
            # Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
            
            if st[0][0] == 1:
                # Valid flow
                new_center = p1[0][0]
                dx = new_center[0] - prev_pts[0][0][0]
                dy = new_center[1] - prev_pts[0][0][1]
                
                # Update bbox with flow
                if prev_bbox is not None:
                    x1, y1, x2, y2 = prev_bbox
                    w, h = x2 - x1, y2 - y1
                    
                    # Constrain movement (outlier rejection)
                    if abs(dx) < 50 and abs(dy) < 50:
                        nx1 = int(x1 + dx)
                        ny1 = int(y1 + dy)
                        nx2 = nx1 + w
                        ny2 = ny1 + h
                        
                        # Bounds check
                        h_img, w_img = frame_gray.shape
                        nx1 = max(0, min(w_img-1, nx1))
                        ny1 = max(0, min(h_img-1, ny1))
                        nx2 = max(0, min(w_img-1, nx2))
                        ny2 = max(0, min(h_img-1, ny2))
                        
                        bbox = np.array([nx1, ny1, nx2, ny2])
                        detected = True # Mark as "detected" (tracked)
                        is_optical_flow = True
                        confidence = 0.5 # Placeholder confidence for flow
        
        # 3. Process Result
        contact_2d = None
        contact_3d = (0.0, 0.0, 0.0)
        depth_val = 0.0
        depth_std = 0.0
        
        if detected and bbox is not None:
            x1, y1, x2, y2 = bbox
            contact_u = (x1 + x2) // 2
            contact_v = y1
            contact_2d = [int(contact_u), int(contact_v)]
            
            # Update Optical Flow points for next frame
            prev_pts = np.array([[[float(contact_u), float(contact_v)]]], dtype=np.float32)
            prev_bbox = bbox
            
            # Load depth and estimate 3D
            if depth_path.exists():
                depth = _load_depth(depth_path)
                if depth is not None:
                    x, y, z, d, d_std = _estimate_contact_3d(bbox, depth, intrinsics)
                    contact_3d = (x, y, z)
                    depth_val = d
                    depth_std = d_std
        else:
            # Reset flow if tracking lost completely
            prev_pts = None
            prev_bbox = None
            
        # Store track
        tracks.append({
            "frame": name,
            "detected": detected,
            "is_optical_flow": is_optical_flow,
            "confidence": confidence,
            "bbox": [int(b) for b in bbox] if bbox is not None else None,
            "contact_2d": contact_2d,
            "contact_3d": list(contact_3d),
            "depth_val": depth_val,
            "depth_std": depth_std
        })
        
        # Update previous frame
        prev_gray = frame_gray.copy()
        
        # Visualize
        if vis_dir and frame_img is not None:
            if detected and bbox is not None:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 255) if is_optical_flow else (0, 255, 0) # Yellow for Flow, Green for YOLO
                
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
                if contact_2d:
                    cv2.circle(frame_img, tuple(contact_2d), 5, (0, 0, 255), -1)
                
                label = "Flow" if is_optical_flow else f"YOLO {confidence:.2f}"
                text = f"{label} | 3D: ({contact_3d[0]:.2f}, {contact_3d[1]:.2f}, {contact_3d[2]:.2f})"
                cv2.putText(frame_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imwrite(str(vis_dir / f"{name}.jpg"), frame_img)
    
    # Compute statistics
    detected_count = sum(1 for t in tracks if t["detected"])
    flow_count = sum(1 for t in tracks if t.get("is_optical_flow", False))
    yolo_count = detected_count - flow_count
    
    avg_conf = np.mean([t["confidence"] for t in tracks if t["detected"]]) if detected_count > 0 else 0.0
    
    # Compute velocities
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
            "yolo_detections": yolo_count,
            "optical_flow_detections": flow_count,
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
    print(f"    - YOLO: {yolo_count}")
    print(f"    - Optical Flow: {flow_count}")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"  Saved to: {output_json}")
    
    if vis_dir:
        print(f"  Visualizations: {vis_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
