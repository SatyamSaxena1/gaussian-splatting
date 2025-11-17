"""Pole detection utilities.

This package provides helpers to detect pole-like cylindrical structures
from point clouds exported by 3D Gaussian Splatting training runs.
"""

from .geometry import (
    axis_endpoints,
    cuboid_faces,
    cuboid_vertices,
    cuboid_wire_segments,
    extract_detection_arrays,
    orthonormal_basis,
)
from .ransac import CylinderModel, load_point_cloud, ransac_cylinder
from .synthetic import CylinderSpec, generate_cylinder_point_cloud, save_point_cloud_ply

from .detect import (
    DetectionResult,
    detect_multiple_poles_from_file,
    detect_multiple_poles_from_points,
    detect_pole_from_file,
    detect_pole_from_points,
    detection_to_dict,
    detections_to_dicts,
    save_detection_json,
)

__all__ = [
    "CylinderModel",
    "CylinderSpec",
    "generate_cylinder_point_cloud",
    "save_point_cloud_ply",
    "load_point_cloud",
    "ransac_cylinder",
    "DetectionResult",
    "detect_multiple_poles_from_file",
    "detect_multiple_poles_from_points",
    "detect_pole_from_file",
    "detect_pole_from_points",
    "detection_to_dict",
    "detections_to_dicts",
    "save_detection_json",
    "axis_endpoints",
    "cuboid_faces",
    "cuboid_vertices",
    "cuboid_wire_segments",
    "extract_detection_arrays",
    "orthonormal_basis",
]
