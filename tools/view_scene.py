#!/usr/bin/env python3
"""Simple scene visualizer.

Usage:
  python3 tools/view_scene.py --model output/2de404ca-3 --iteration 30000 --mode html

Creates output/{model}/visualization.html containing an interactive Plotly 3D scatter of the point cloud
and camera frusta (if cameras.json is present).
"""
import argparse
import os
import json
import numpy as np

def read_ply_positions(path):
    # Minimal PLY reader for vertex positions and colors
    try:
        from plyfile import PlyData
    except Exception:
        raise RuntimeError("plyfile package is required for reading PLY. Install with pip install plyfile")
    ply = PlyData.read(path)
    vertices = ply['vertex']
    xyz = np.vstack([vertices[t] for t in ('x','y','z')]).T
    try:
        colors = np.vstack([vertices[t] for t in ('red','green','blue')]).T / 255.0
    except Exception:
        colors = None
    return xyz, colors

def load_cameras_json(path):
    with open(path, 'r') as f:
        cams = json.load(f)
    # Expect list of dicts with transform/extrinsics saved by the pipeline
    return cams

def make_plotly_html(points, colors, cameras, outpath, max_points=200000):
    try:
        import plotly.graph_objects as go
    except Exception:
        raise RuntimeError("plotly is required to generate HTML visualization. Install with pip install plotly")

    # Subsample points if too many
    n = points.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        pts = points[idx]
        cols = colors[idx] if (colors is not None) else None
    else:
        pts = points
        cols = colors

    marker = dict(size=1)
    if cols is not None:
        marker['color'] = (cols * 255).astype(int).tolist()
        marker['colorscale'] = None
        marker['showscale'] = False

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=marker, name='points'))

    # Add simple camera markers if available (cameras.json expected format: list of dicts with 'center' and 'name')
    if cameras:
        cam_centers = []
        cam_text = []
        for cam in cameras:
            try:
                # expect camera JSON produced earlier has 'center' (list) or parse from JSON depending on format
                center = cam.get('center', None) or cam.get('position', None) or cam.get('cam_center', None)
                if center is None:
                    # try heuristic: cameras.json in repo uses camera entries with 'R' and 'T' maybe; skip
                    continue
                cam_centers.append(center)
                cam_text.append(cam.get('image_name', cam.get('name', 'cam')))
            except Exception:
                continue
        if cam_centers:
            cc = np.array(cam_centers)
            fig.add_trace(go.Scatter3d(x=cc[:,0], y=cc[:,1], z=cc[:,2], mode='markers+text', marker=dict(size=4, color='red'), text=cam_text, name='cameras'))

    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,b=0,t=0))
    fig.write_html(outpath, include_plotlyjs='cdn')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to output model folder')
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--mode', choices=['html'], default='html')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    model_path = args.model
    if args.iteration and args.iteration > 0:
        ply_path = os.path.join(model_path, 'point_cloud', f'iteration_{args.iteration}', 'point_cloud.ply')
    else:
        ply_path = os.path.join(model_path, 'point_cloud', 'iteration_30000', 'point_cloud.ply')

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    points, colors = read_ply_positions(ply_path)

    cams_json = os.path.join(model_path, 'cameras.json')
    cameras = None
    if os.path.exists(cams_json):
        cameras = load_cameras_json(cams_json)

    outpath = args.out or os.path.join(model_path, 'visualization.html')
    make_plotly_html(points, colors, cameras, outpath)
    print('Wrote', outpath)

if __name__ == '__main__':
    main()
