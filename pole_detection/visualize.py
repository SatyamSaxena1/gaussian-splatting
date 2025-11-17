"""Visualization helpers for pole detection results.

This script renders a point cloud together with the detected pole and an
oriented bounding cuboid so that the geometry can be inspected inside the
Gaussian Splatting output coordinate frame.

Example
-------
.. code-block:: bash

	MPLBACKEND=Agg python3 -m pole_detection.visualize \
		output/scene/point_cloud/iteration_30000/point_cloud.ply \
		output/scene/pole_detection.json \
		--output output/scene/pole_visualization.png

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import axis_endpoints, cuboid_vertices, extract_detection_arrays
from .ransac import load_point_cloud


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Visualise pole detection results with a bounding cuboid",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("point_cloud", type=Path, help="Path to the input point cloud (.ply)")
	parser.add_argument("detection", type=Path, help="JSON file produced by the detection CLI")
	parser.add_argument(
		"--output",
		type=Path,
		help="Optional path to save the rendered figure instead of showing it",
	)
	parser.add_argument(
		"--max-points",
		type=int,
		default=60000,
		help="Randomly subsample to this many points for plotting (0 disables)",
	)
	parser.add_argument(
		"--radius-tolerance",
		type=float,
		default=0.05,
		help="Extra radial tolerance (in scene units) when marking inliers",
	)
	parser.add_argument(
		"--radius-scale",
		type=float,
		default=1.2,
		help="Scale applied to the detected radius when building the cuboid",
	)
	parser.add_argument(
		"--point-size",
		type=float,
		default=1.0,
		help="Scatter point size for matplotlib",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed used for plot subsampling",
	)
	return parser


def main(argv: Optional[list[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	if not args.point_cloud.exists():
		parser.error(f"Point cloud file '{args.point_cloud}' does not exist")

	if not args.detection.exists():
		parser.error(f"Detection file '{args.detection}' does not exist")

	point_cloud = load_point_cloud(str(args.point_cloud))

	with open(args.detection, "r", encoding="utf8") as fp:
		detection_raw = json.load(fp)

	detections = detection_raw if isinstance(detection_raw, list) else [detection_raw]
	if not detections:
		parser.error("Detection JSON contains no entries")

	required_keys = {"axis_point", "axis_direction", "radius", "axial_bounds", "height"}
	for idx, det in enumerate(detections):
		missing_keys = required_keys - det.keys()
		if missing_keys:
			parser.error(
				f"Detection JSON entry {idx} missing required keys: {sorted(missing_keys)}"
			)

	rng = np.random.default_rng(args.seed)
	if args.max_points and point_cloud.shape[0] > args.max_points:
		idx = rng.choice(point_cloud.shape[0], size=args.max_points, replace=False)
		point_cloud = point_cloud[idx]

	inlier_mask = np.zeros(point_cloud.shape[0], dtype=bool)
	for det in detections:
		inlier_mask |= compute_inlier_mask(point_cloud, det, args.radius_tolerance)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection="3d")

	plot_point_cloud(ax, point_cloud, inlier_mask, args.point_size)
	palette = [
		"#1a237e",
		"#c62828",
		"#2e7d32",
		"#6a1b9a",
		"#ff8f00",
	]

	for idx, det in enumerate(detections):
		colour = palette[idx % len(palette)]
		label = "Cylinder axis" if idx == 0 else None
		plot_axis(ax, det, colour, label=label)
		plot_cuboid(ax, det, colour, radius_scale=args.radius_scale, alpha=0.18)

	configure_axes(ax, point_cloud)
	fig.tight_layout()

	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(args.output, dpi=200)
	else:
		plt.show()

	return 0


def compute_inlier_mask(points: np.ndarray, detection: dict, tolerance: float) -> np.ndarray:
	axis_point, axis_dir, _, radius = extract_detection_arrays(detection)
	rel = points - axis_point
	axial_projection = rel @ axis_dir
	projection_points = axis_point + np.outer(axial_projection, axis_dir)
	radial_dist = np.linalg.norm(points - projection_points, axis=1)
	return np.abs(radial_dist - radius) <= tolerance


def plot_point_cloud(ax, points: np.ndarray, inliers: np.ndarray, point_size: float) -> None:
	ax.scatter(
		points[~inliers, 0],
		points[~inliers, 1],
		points[~inliers, 2],
		s=point_size,
		c="#b0b0b0",
		alpha=0.15,
		linewidths=0,
	)

	ax.scatter(
		points[inliers, 0],
		points[inliers, 1],
		points[inliers, 2],
		s=point_size * 2.0,
		c="#ff6f00",
		alpha=0.6,
		linewidths=0,
	)


def plot_axis(ax, detection: dict, color: str, label: Optional[str] = None) -> None:
	bottom, top = axis_endpoints(detection)

	ax.plot(
		[bottom[0], top[0]],
		[bottom[1], top[1]],
		[bottom[2], top[2]],
		color=color,
		linewidth=2.0,
		label=label,
	)


def plot_cuboid(ax, detection: dict, color: str, *, radius_scale: float = 1.2, alpha: float = 0.15) -> None:
	vertices = cuboid_vertices(detection, radius_scale=radius_scale)

	faces = [
		[vertices[i] for i in [0, 1, 2, 3]],  # bottom
		[vertices[i] for i in [4, 5, 6, 7]],  # top
		[vertices[i] for i in [0, 1, 5, 4]],
		[vertices[i] for i in [1, 2, 6, 5]],
		[vertices[i] for i in [2, 3, 7, 6]],
		[vertices[i] for i in [3, 0, 4, 7]],
	]

	poly = Poly3DCollection(faces, alpha=alpha, linewidths=1.0)
	poly.set_facecolor(color)
	poly.set_edgecolor(color)
	ax.add_collection3d(poly)


def configure_axes(ax, points: np.ndarray) -> None:
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.set_box_aspect((
		np.ptp(points[:, 0]),
		np.ptp(points[:, 1]),
		np.ptp(points[:, 2]),
	))
	ax.view_init(elev=20.0, azim=-60.0)
	ax.grid(False)
	ax.legend(loc="upper right")


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
