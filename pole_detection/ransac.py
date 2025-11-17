"""RANSAC-based pole detection utilities.

This module implements the first building block for detecting pole-like
structures from a point cloud. The core routine runs a lightweight
RANSAC loop that repeatedly samples subsets of points, estimates a
candidate cylinder with principal component analysis, and scores it
against the full point set.

The implementation intentionally avoids heavyweight dependencies so that
it can run inside the existing Gaussian Splatting environment. Only
``numpy`` and ``plyfile`` are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData


@dataclass
class CylinderModel:
    """Represents a fitted cylinder model.

    Attributes
    ----------
    axis_point:
        A point on the cylinder axis (centroid of the inlier points).
    axis_direction:
        Unit-length vector representing the cylinder axis direction.
    radius:
        Estimated radius of the cylinder.
    height:
        Extent of the inlier points along the axis direction.
    inlier_mask:
        Boolean mask over the original point set marking the inliers.
    axial_bounds:
        Tuple containing the minimum and maximum signed distances of the
        inlier points along the cylinder axis relative to ``axis_point``.
    """

    axis_point: np.ndarray
    axis_direction: np.ndarray
    radius: float
    height: float
    inlier_mask: np.ndarray
    axial_bounds: Tuple[float, float]

    def inlier_count(self) -> int:
        """Return the number of inliers supporting this model."""

        return int(np.count_nonzero(self.inlier_mask))

    def endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return points corresponding to the bottom and top of the cylinder."""

        bottom = self.axis_point + self.axial_bounds[0] * self.axis_direction
        top = self.axis_point + self.axial_bounds[1] * self.axis_direction
        return bottom, top


def load_point_cloud(path: str, fields: Sequence[str] = ("x", "y", "z")) -> np.ndarray:
    """Load a point cloud from a PLY file.

    Parameters
    ----------
    path:
        Path to the ``.ply`` file containing a vertex element.
    fields:
        Names of the vertex fields to extract. Defaults to the ``x``,
        ``y``, and ``z`` coordinates.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(N, len(fields))`` containing the requested
        vertex attributes as ``float32``.

    Raises
    ------
    ValueError
        If any requested field is missing from the PLY file.
    """

    ply = PlyData.read(path)
    try:
        vertex_data = ply["vertex"].data
    except KeyError as exc:
        raise ValueError(
            f"PLY file '{path}' does not contain a vertex element"
        ) from exc
    missing = [field for field in fields if field not in vertex_data.dtype.names]
    if missing:
        raise ValueError(
            f"PLY file '{path}' is missing required vertex fields: {missing}"
        )

    stacked = np.vstack([vertex_data[field] for field in fields]).T
    return stacked.astype(np.float32, copy=False)


def ransac_cylinder(
    points: np.ndarray,
    num_iterations: int = 256,
    sample_size: int = 512,
    radial_threshold: float = 0.05,
    min_inliers: int = 500,
    min_height: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Optional[CylinderModel]:
    """Fit a cylinder to the provided points using a simple RANSAC loop.

    Parameters
    ----------
    points:
        Array of shape ``(N, 3)`` containing the input point cloud.
    num_iterations:
        Number of RANSAC iterations to perform.
    sample_size:
        Number of points to sample when generating a candidate model.
    radial_threshold:
        Maximum absolute deviation from the estimated radius for a point
        to be considered an inlier (same units as the point cloud).
    min_inliers:
        Minimum number of inliers required for a model to be accepted.
    min_height:
        Optional minimum axial height the cylinder must span. If ``None``
        the height constraint is skipped.
    random_state:
        Seed for the internal random generator.

    Returns
    -------
    Optional[CylinderModel]
        The best cylinder model found, or ``None`` if no satisfactory
        model could be estimated.
    """

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an array with shape (N, 3)")

    num_points = points.shape[0]
    if num_points < max(3, sample_size):
        sample_size = num_points

    rng = np.random.default_rng(random_state)
    best_model: Optional[CylinderModel] = None
    best_inlier_count = 0

    for _ in range(num_iterations):
        if num_points < 3:
            break

        subset_indices = rng.choice(num_points, size=sample_size, replace=False)
        subset = points[subset_indices]

        candidate = _fit_cylinder_from_subset(subset)
        if candidate is None:
            continue

        axis_point, axis_dir, radius = candidate
        model = _score_model(
            points,
            axis_point,
            axis_dir,
            radius,
            radial_threshold=radial_threshold,
            min_inliers=min_inliers,
            min_height=min_height,
        )

        if model is None:
            continue

        inlier_count = model.inlier_count()
        if inlier_count > best_inlier_count:
            best_model = model
            best_inlier_count = inlier_count

    return best_model


def _fit_cylinder_from_subset(subset: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Estimate a cylinder axis and radius from a subset of points."""

    centroid, axis_dir = _principal_axis(subset)
    if centroid is None or axis_dir is None:
        return None

    radius = _estimate_radius(subset, centroid, axis_dir)
    if not np.isfinite(radius) or radius <= 0:
        return None

    return centroid, axis_dir, float(radius)


def _score_model(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius: float,
    *,
    radial_threshold: float,
    min_inliers: int,
    min_height: Optional[float],
) -> Optional[CylinderModel]:
    """Compute inliers for a candidate cylinder and optionally refine it."""

    radial_dist = _radial_distances(points, axis_point, axis_dir)
    inlier_mask = np.abs(radial_dist - radius) <= radial_threshold
    inlier_count = int(np.count_nonzero(inlier_mask))

    if inlier_count < max(3, min_inliers):
        return None

    refined = _refine_model(points[inlier_mask])
    if refined is None:
        return None

    axis_point_refined, axis_dir_refined, radius_refined, axial_bounds = refined

    radial_dist_refined = _radial_distances(points, axis_point_refined, axis_dir_refined)
    refined_mask = np.abs(radial_dist_refined - radius_refined) <= radial_threshold
    refined_count = int(np.count_nonzero(refined_mask))

    if refined_count < max(3, min_inliers):
        return None

    height = axial_bounds[1] - axial_bounds[0]
    if min_height is not None and height < min_height:
        return None

    return CylinderModel(
        axis_point=axis_point_refined,
        axis_direction=axis_dir_refined,
        radius=float(radius_refined),
        height=float(height),
        inlier_mask=refined_mask,
        axial_bounds=(float(axial_bounds[0]), float(axial_bounds[1])),
    )


def _principal_axis(points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute the principal axis of the supplied points via PCA."""

    if points.size == 0:
        return None, None

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    if centered.shape[0] < 3:
        return centroid, _normalize(centered[0] if centered.shape[0] > 0 else np.array([1.0, 0.0, 0.0]))

    covariance = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    principal_idx = int(np.argmax(eigvals))
    axis_dir = eigvecs[:, principal_idx]
    axis_dir = _normalize(axis_dir)

    if axis_dir is None:
        return None, None

    return centroid, axis_dir


def _estimate_radius(points: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> float:
    """Estimate the cylinder radius as the median radial distance."""

    radial_distances = _radial_distances(points, axis_point, axis_dir)
    return float(np.median(radial_distances))


def _radial_distances(points: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Compute distances from points to the cylinder axis."""

    axis_dir = _normalize(axis_dir)
    rel = points - axis_point
    axial_projection = np.dot(rel, axis_dir)
    projection_points = axis_point + np.outer(axial_projection, axis_dir)
    return np.linalg.norm(points - projection_points, axis=1)


def _refine_model(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]]:
    """Refine the cylinder parameters using all inlier points."""

    if points.shape[0] < 3:
        return None

    centroid, axis_dir = _principal_axis(points)
    if centroid is None or axis_dir is None:
        return None

    radius = _estimate_radius(points, centroid, axis_dir)
    if not np.isfinite(radius) or radius <= 0:
        return None

    axial_positions = _axial_positions(points, centroid, axis_dir)
    axial_min = float(np.min(axial_positions))
    axial_max = float(np.max(axial_positions))

    return centroid, axis_dir, radius, (axial_min, axial_max)


def _axial_positions(points: np.ndarray, axis_point: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Project points onto the cylinder axis."""

    axis_dir = _normalize(axis_dir)
    rel = points - axis_point
    return np.dot(rel, axis_dir)


def _normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    """Return a unit-length copy of ``vector`` or ``None`` if zero-length."""

    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0.0:
        return None
    return vector / norm
