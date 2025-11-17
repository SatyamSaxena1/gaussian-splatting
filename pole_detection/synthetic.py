"""Synthetic data utilities for pole detection tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass
class CylinderSpec:
    """Defines the geometry of a synthetic cylinder."""

    center: np.ndarray
    axis_direction: np.ndarray
    radius: float
    height: float


def generate_cylinder_point_cloud(
    spec: CylinderSpec,
    num_points: int,
    noise_std: float = 0.01,
    outlier_ratio: float = 0.05,
    outlier_bounds: Tuple[float, float] = (-2.0, 2.0),
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate a noisy point cloud of a cylinder with optional outliers."""

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if not (0.0 <= outlier_ratio < 1.0):
        raise ValueError("outlier_ratio must be in [0, 1)")

    rng = np.random.default_rng(random_state)

    axis_dir = spec.axis_direction / np.linalg.norm(spec.axis_direction)

    n_inliers = int(num_points * (1.0 - outlier_ratio))
    n_outliers = num_points - n_inliers

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_inliers)
    axial_positions = rng.uniform(-spec.height / 2.0, spec.height / 2.0, size=n_inliers)

    a, b = _orthonormal_basis(axis_dir)
    circle_points = (
        spec.radius * np.cos(angles)[:, None] * a
        + spec.radius * np.sin(angles)[:, None] * b
    )
    axial_vector = axial_positions[:, None] * axis_dir
    inliers = spec.center + circle_points + axial_vector
    inliers += rng.normal(scale=noise_std, size=inliers.shape)

    if n_outliers > 0:
        low, high = outlier_bounds
        outliers = rng.uniform(low=low, high=high, size=(n_outliers, 3))
        points = np.vstack([inliers, outliers])
    else:
        points = inliers

    rng.shuffle(points)
    return points.astype(np.float32)


def _orthonormal_basis(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors perpendicular to ``direction``."""

    direction = direction / np.linalg.norm(direction)
    if np.allclose(direction, np.array([0.0, 0.0, 1.0])):
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 0.0, 1.0])

    a = np.cross(direction, arbitrary)
    a /= np.linalg.norm(a)
    b = np.cross(direction, a)
    b /= np.linalg.norm(b)
    return a, b


def save_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    """Write a point cloud array to a binary little-endian PLY file."""

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    vertex_data = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]

    element = PlyElement.describe(vertex_data, "vertex")
    ply = PlyData([element], text=False)
    ply.write(str(path))
