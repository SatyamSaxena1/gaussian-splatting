#!/usr/bin/env python3
"""Helper to export a nerfvis scene for the trained model.

Usage:
  python3 tools/run_nerfvis.py --model output/2de404ca-3 --iteration 30000 \
      --outdir output/2de404ca-3/nerfvis_export

This script will create a directory containing a static nerfvis export you can
open in a browser.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

# Ensure the vendored nerfvis package takes precedence over any installed copy.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENDORED_NERFVIS_ROOT = os.path.join(SCRIPT_DIR, "nerfvis")
if VENDORED_NERFVIS_ROOT not in sys.path:
    sys.path.insert(0, VENDORED_NERFVIS_ROOT)

from pole_detection.geometry import (  # noqa: E402
    axis_endpoints,
    cuboid_faces,
    cuboid_vertices,
    cuboid_wire_segments,
)

def load_ply_xyz(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply['vertex']
    xyz = np.vstack([v['x'], v['y'], v['z']]).T
    try:
        colors = np.vstack([v['red'], v['green'], v['blue']]).T / 255.0
    except Exception:
        colors = None
    return xyz, colors


def available_iterations(model_dir: str) -> List[int]:
    point_cloud_root = os.path.join(model_dir, 'point_cloud')
    if not os.path.isdir(point_cloud_root):
        return []
    pattern = re.compile(r"^iteration_(\d+)$")
    iterations = []
    for entry in os.listdir(point_cloud_root):
        match = pattern.match(entry)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def resolve_point_cloud_path(model_dir: str, requested_iter: Optional[int]) -> Tuple[str, int]:
    iterations = available_iterations(model_dir)
    if not iterations:
        raise FileNotFoundError(
            f"No iteration folders found under '{model_dir}/point_cloud'. Did training finish?"
        )

    def build_path(iteration: int) -> str:
        return os.path.join(model_dir, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')

    if requested_iter is not None:
        candidate = build_path(requested_iter)
        if os.path.exists(candidate):
            return candidate, requested_iter
        # Provide a helpful error mentioning available options.
        available_str = ', '.join(str(it) for it in iterations)
        raise FileNotFoundError(
            f"Point cloud for iteration {requested_iter} not found at '{candidate}'."
            f" Available iterations: {available_str}"
        )

    # Fallback to the highest available iteration when none was specified.
    best_iter = iterations[-1]
    candidate = build_path(best_iter)
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Expected point cloud at '{candidate}' but file is missing despite iteration folder"
        )
    return candidate, best_iter


def parse_hex_color(hex_color: str) -> np.ndarray:
    color = hex_color.strip()
    if color.startswith('#'):
        color = color[1:]
    if len(color) not in (3, 6):
        raise ValueError(f"Color '{hex_color}' must be in #RGB or #RRGGBB form")
    if len(color) == 3:
        color = ''.join(ch * 2 for ch in color)
    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color '{hex_color}'") from exc
    return np.array([r, g, b], dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--iteration', type=int, default=None,
                        help="Optional iteration to export; defaults to highest available")
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--detection', type=str, default=None,
                        help="Optional path to pole detection JSON; defaults to <model>/pole_detection.json")
    parser.add_argument('--radius-scale', type=float, default=1.2,
                        help="Scale factor applied to detected radius when building cuboid")
    parser.add_argument('--bbox-style', choices=['wire', 'solid', 'both'], default='wire',
                        help="Rendering style for pole bounding boxes")
    parser.add_argument('--bbox-color', type=str, default="#00FF00",
                        help="Bounding box colour as #RRGGBB or #RGB")
    parser.add_argument('--bbox-opacity', type=float, default=0.18,
                        help="Opacity for solid bounding boxes (ignored for wireframe)")
    args = parser.parse_args()

    model = args.model
    ply_path, iteration_used = resolve_point_cloud_path(model, args.iteration)

    xyz, cols = load_ply_xyz(ply_path)

    cams_json = os.path.join(model, 'cameras.json')
    cameras = None
    if os.path.exists(cams_json):
        with open(cams_json, 'r') as f:
            cameras = json.load(f)

    if args.detection:
        detection_path = args.detection
    else:
        multi_candidate = os.path.join(model, 'pole_detections.json')
        single_candidate = os.path.join(model, 'pole_detection.json')
        detection_path = multi_candidate if os.path.exists(multi_candidate) else single_candidate
    detection = None
    if detection_path and os.path.exists(detection_path):
        with open(detection_path, 'r', encoding='utf8') as f:
            detection = json.load(f)
    elif args.detection:
        raise FileNotFoundError(f"Detection file '{detection_path}' not found")

    # Import nerfvis and build scene
    import nerfvis
    s = nerfvis.Scene('Trained model')
    # set OpenCV convention (Y up negative) to match repository conventions
    try:
        s.set_opencv()
    except Exception:
        pass

    if cols is not None:
        s.add_points('point_cloud', xyz, vert_color=cols)
    else:
        s.add_points('point_cloud', xyz, color=[0.5, 0.5, 0.5])

    if cameras:
        # Build c2w matrices from rotation + position
        Rs = []
        Ts = []
        for cam in cameras:
            R = np.array(cam.get('rotation'))
            t = np.array(cam.get('position'))
            # Nerfvis expects c2w rotation and translation arrays r,t
            Rs.append(R)
            Ts.append(t)
        Rs = np.stack(Rs, axis=0)
        Ts = np.stack(Ts, axis=0)
        # Use fx from camera entry (approx focal in pixels)
        f = cameras[0].get('fx', None) or cameras[0].get('fy', None)
        s.add_images('cams', [], r=Rs, t=Ts, focal_length=float(f) if f else None, with_camera_frustum=True, z=0.5)

    if detection is not None:
        detections = detection if isinstance(detection, list) else [detection]
        try:
            bbox_color = parse_hex_color(args.bbox_color)
        except ValueError as exc:
            raise RuntimeError(f"Invalid bbox color '{args.bbox_color}': {exc}") from exc

        style = args.bbox_style
        axis_color = np.array([1.0, 0.25, 0.25], dtype=np.float32)
        style_label = {
            'wire': 'wireframe',
            'solid': 'solid',
            'both': 'solid + wireframe',
        }[style]

        faces = cuboid_faces()
        segs_template = cuboid_wire_segments()

        for idx, det in enumerate(detections):
            try:
                cuboid = cuboid_vertices(det, radius_scale=args.radius_scale)
                segs = segs_template
                bottom, top = axis_endpoints(det)
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"Failed to interpret detection JSON '{detection_path}': {exc}") from exc

            suffix = f"_{idx+1}" if len(detections) > 1 else ""
            solid_name = f"pole_cuboid{suffix}"
            wire_name = f"pole_cuboid_wire{suffix}"
            axis_name = f"pole_axis{suffix}"

            if style in ('solid', 'both'):
                opacity = np.clip(args.bbox_opacity, 0.0, 1.0)
                solid_color = bbox_color * opacity + np.array([1.0, 1.0, 1.0], dtype=np.float32) * (1.0 - opacity)
                s.add_mesh(solid_name, cuboid, faces=faces, face_size=3, color=solid_color.tolist(), unlit=True)

            if style in ('wire', 'both'):
                s.add_lines(
                    wire_name,
                    cuboid,
                    segs=segs,
                    color=bbox_color.tolist(),
                    update_bb=False,
                    unlit=True,
                )

            s.add_lines(
                axis_name,
                np.stack([bottom, top], axis=0),
                segs=np.array([[0, 1]], dtype=np.int32),
                color=axis_color,
                update_bb=False,
                unlit=True,
            )

        print(
            f"Added {len(detections)} pole bounding box{'es' if len(detections) != 1 else ''} "
            f"({style_label}, color={args.bbox_color.upper()})"
        )

    outdir = args.outdir or os.path.join(model, 'nerfvis_export')
    os.makedirs(outdir, exist_ok=True)
    print(f'Exporting nerfvis scene for iteration {iteration_used} to {outdir}')
    s.export(outdir)
    print('Done')

if __name__ == '__main__':
    main()
