"""High-level detection helpers for pole-like structures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .ransac import CylinderModel, load_point_cloud, ransac_cylinder

UP_VECTOR = np.array([0.0, 0.0, 1.0], dtype=np.float32)


@dataclass
class DetectionResult:
    """Container for a detected pole-like cylinder."""

    model: CylinderModel
    total_points: int
    lean_angle_degrees: float

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the detection."""

        bottom, top = self.model.endpoints()
        inlier_count = self.model.inlier_count()
        return {
            "axis_point": self.model.axis_point.tolist(),
            "axis_direction": self.model.axis_direction.tolist(),
            "radius": float(self.model.radius),
            "height": float(self.model.height),
            "axial_bounds": list(self.model.axial_bounds),
            "lean_angle_degrees": float(self.lean_angle_degrees),
            "inlier_count": int(inlier_count),
            "total_points": int(self.total_points),
            "inlier_ratio": float(inlier_count / max(1, self.total_points)),
            "bottom_endpoint": bottom.tolist(),
            "top_endpoint": top.tolist(),
        }


def detection_to_dict(result: DetectionResult) -> dict:
    """Return a dictionary representation of a detection result."""

    return result.to_dict()


def detections_to_dicts(results: Sequence[DetectionResult]) -> List[dict]:
    """Serialise multiple detections to dictionaries."""

    return [res.to_dict() for res in results]


def detect_pole_from_points(
    points: np.ndarray,
    *,
    num_iterations: int = 512,
    sample_size: int = 1024,
    radial_threshold: float = 0.05,
    min_inliers: int = 800,
    min_height: Optional[float] = 1.0,
    max_points: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Optional[DetectionResult]:
    """Run cylinder detection on an in-memory point cloud."""

    points = np.asarray(points, dtype=np.float32)
    total_points = points.shape[0]

    if max_points is not None and total_points > max_points:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(total_points, size=max_points, replace=False)
        points = points[indices]

    model = ransac_cylinder(
        points,
        num_iterations=num_iterations,
        sample_size=min(sample_size, points.shape[0]),
        radial_threshold=radial_threshold,
        min_inliers=min_inliers,
        min_height=min_height,
        random_state=random_state,
    )

    if model is None:
        return None

    lean = _lean_angle_degrees(model.axis_direction, UP_VECTOR)
    return DetectionResult(model=model, total_points=total_points, lean_angle_degrees=lean)


def detect_multiple_poles_from_points(
    points: np.ndarray,
    *,
    max_poles: Optional[int] = None,
    removal_padding: float = 0.02,
    random_state: Optional[int] = None,
    **kwargs,
) -> List[DetectionResult]:
    """Iteratively detect multiple poles by peeling inliers from the cloud.

    Parameters mirror :func:`detect_pole_from_points`. ``max_poles`` limits the
    number of returned detections (``None`` for no limit). ``removal_padding``
    expands the radial and axial envelopes slightly when removing inliers to
    avoid leaving behind fringe points that belong to the detected pole.
    """

    points = np.asarray(points, dtype=np.float32)
    remaining = points.copy()

    # Ensure we never downsample when peeling, otherwise the inlier mask would
    # no longer align with the remaining point set. Users can still constrain
    # the RANSAC search through ``sample_size`` and ``num_iterations``.
    single_kwargs = dict(kwargs)
    single_kwargs.pop("random_state", None)
    max_points_param = single_kwargs.pop("max_points", None)
    if max_points_param in (None, 0):
        max_points_param = None

    radial_threshold = float(single_kwargs.get("radial_threshold", 0.05))
    min_inliers = int(single_kwargs.get("min_inliers", 800))

    rng = np.random.default_rng(random_state) if random_state is not None else None

    detections: List[DetectionResult] = []
    iteration = 0
    while remaining.shape[0] >= max(3, min_inliers):
        iteration += 1
        seed = None
        if rng is not None:
            seed = int(rng.integers(0, 2**31 - 1))

        result = detect_pole_from_points(
            remaining,
            random_state=seed,
            max_points=max_points_param,
            **single_kwargs,
        )

        if result is None:
            break

        detections.append(result)

        mask = _cylinder_membership_mask(
            remaining,
            result.model,
            radial_threshold=radial_threshold,
            padding=removal_padding,
        )

        if not np.any(mask):  # Defensive guard; should not occur.
            break

        remaining = remaining[~mask]

        if max_poles is not None and len(detections) >= max_poles:
            break

    return detections


def detect_pole_from_file(
    path: Path | str,
    *,
    fields: Sequence[str] = ("x", "y", "z"),
    **kwargs,
) -> Optional[DetectionResult]:
    """Load a PLY point cloud and detect the most prominent pole."""

    points = load_point_cloud(str(path), fields)
    return detect_pole_from_points(points, **kwargs)


def detect_multiple_poles_from_file(
    path: Path | str,
    *,
    fields: Sequence[str] = ("x", "y", "z"),
    **kwargs,
) -> List[DetectionResult]:
    """Load a PLY point cloud and iteratively detect multiple poles."""

    points = load_point_cloud(str(path), fields)
    return detect_multiple_poles_from_points(points, **kwargs)


def save_detection_json(path: Path | str, result: DetectionResult | Sequence[DetectionResult]) -> None:
    """Persist one or more detection results to JSON."""

    if isinstance(result, DetectionResult):
        payload: Iterable[dict] | dict = result.to_dict()
    else:
        payload = [res.to_dict() for res in result]

    with open(path, "w", encoding="utf8") as fp:
        json.dump(payload, fp, indent=2)


def _lean_angle_degrees(axis_direction: np.ndarray, up_vector: np.ndarray) -> float:
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    up_vector = up_vector / np.linalg.norm(up_vector)
    dot = float(np.clip(np.abs(np.dot(axis_direction, up_vector)), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _cylinder_membership_mask(
    points: np.ndarray,
    model: CylinderModel,
    *,
    radial_threshold: float,
    padding: float,
) -> np.ndarray:
    """Return a boolean mask selecting points belonging to ``model``."""

    axis_dir = model.axis_direction / np.linalg.norm(model.axis_direction)
    rel = points - model.axis_point
    axial_positions = rel @ axis_dir
    axial_min, axial_max = model.axial_bounds
    axial_mask = (axial_positions >= axial_min - padding) & (axial_positions <= axial_max + padding)

    projection = model.axis_point + np.outer(axial_positions, axis_dir)
    radial_dist = np.linalg.norm(points - projection, axis=1)
    radial_mask = np.abs(radial_dist - model.radius) <= (radial_threshold + padding)

    return axial_mask & radial_mask
