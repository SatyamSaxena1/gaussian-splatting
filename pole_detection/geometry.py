"""Reusable geometry utilities for pole detection outputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np

DetectionLike = Mapping[str, Any] | Any


def _get_field(detection: DetectionLike, key: str) -> Any:
    if isinstance(detection, Mapping):
        return detection[key]
    if hasattr(detection, key):
        return getattr(detection, key)
    raise KeyError(f"Detection object does not provide field '{key}'")


def _as_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("Cannot normalise zero-length vector")
    return vec / norm


def extract_detection_arrays(
    detection: DetectionLike,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], float]:
    """Return core detection fields as numpy arrays."""

    axis_point = _as_vector(_get_field(detection, "axis_point"))
    axis_direction = _normalise(_get_field(detection, "axis_direction"))
    axial_bounds_raw = _get_field(detection, "axial_bounds")
    if isinstance(axial_bounds_raw, np.ndarray):
        axial_bounds = float(axial_bounds_raw[0]), float(axial_bounds_raw[1])
    else:
        axial_bounds = tuple(float(v) for v in axial_bounds_raw)
    radius = float(_get_field(detection, "radius"))
    return axis_point, axis_direction, (axial_bounds[0], axial_bounds[1]), radius


def axis_endpoints(detection: DetectionLike) -> Tuple[np.ndarray, np.ndarray]:
    axis_point, axis_dir, (axial_min, axial_max), _ = extract_detection_arrays(detection)
    bottom = axis_point + axial_min * axis_dir
    top = axis_point + axial_max * axis_dir
    return bottom, top


def orthonormal_basis(axis_dir: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis_dir = _normalise(axis_dir)
    canonical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.allclose(axis_dir, canonical):
        arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        arbitrary = canonical

    basis_u = arbitrary - axis_dir * np.dot(axis_dir, arbitrary)
    norm_u = np.linalg.norm(basis_u)
    if norm_u == 0.0:
        basis_u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        basis_u = basis_u / norm_u

    basis_v = np.cross(axis_dir, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)
    return basis_u.astype(np.float32), basis_v.astype(np.float32)


def oriented_square(
    center: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    half_extent: float,
) -> np.ndarray:
    offsets = (-basis_u - basis_v, basis_u - basis_v, basis_u + basis_v, -basis_u + basis_v)
    return np.stack([center + half_extent * offset for offset in offsets], axis=0)


def cuboid_vertices(
    detection: DetectionLike,
    radius_scale: float = 1.0,
) -> np.ndarray:
    axis_point, axis_dir, (axial_min, axial_max), radius = extract_detection_arrays(detection)
    scaled_radius = radius * float(radius_scale)
    basis_u, basis_v = orthonormal_basis(axis_dir)

    bottom_center = axis_point + axial_min * axis_dir
    top_center = axis_point + axial_max * axis_dir

    bottom = oriented_square(bottom_center, basis_u, basis_v, scaled_radius)
    top = oriented_square(top_center, basis_u, basis_v, scaled_radius)
    return np.vstack([bottom, top])


def cuboid_faces() -> np.ndarray:
    """Return triangular faces for an axis-aligned cuboid (12 x 3 indices)."""
    return np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )


def cuboid_wire_segments() -> np.ndarray:
    """Return segments for rendering the cuboid as a wireframe (12 x 2 indices)."""
    return np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )