"""Unit tests for the RANSAC cylinder fitting module."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pole_detection.ransac import CylinderModel, ransac_cylinder
from pole_detection.synthetic import CylinderSpec, generate_cylinder_point_cloud


class TestRansacCylinder(unittest.TestCase):
    def test_detects_noisy_cylinder(self) -> None:
        spec = CylinderSpec(
            center=np.array([0.2, -0.1, 0.3], dtype=np.float32),
            axis_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            radius=0.12,
            height=2.0,
        )
        points = generate_cylinder_point_cloud(
            spec,
            num_points=4000,
            noise_std=0.005,
            outlier_ratio=0.1,
            random_state=42,
        )

        model = ransac_cylinder(
            points,
            num_iterations=300,
            sample_size=512,
            radial_threshold=0.02,
            min_inliers=800,
            min_height=1.5,
            random_state=1234,
        )

        self.assertIsInstance(model, CylinderModel)
        assert model is not None  # for type checkers

        recovered_radius = model.radius
        radius_error = abs(recovered_radius - spec.radius)
        self.assertLess(radius_error, 0.02)

        direction_alignment = np.abs(np.dot(model.axis_direction, spec.axis_direction))
        self.assertGreater(direction_alignment, math.cos(math.radians(5)))

        height_error = abs(model.height - spec.height)
        self.assertLess(height_error, 0.4)

        inlier_ratio = model.inlier_count() / points.shape[0]
        self.assertGreater(inlier_ratio, 0.6)

    def test_returns_none_when_points_degenerate(self) -> None:
        points = np.zeros((10, 3), dtype=np.float32)
        model = ransac_cylinder(points, num_iterations=50, sample_size=8)
        self.assertIsNone(model)


if __name__ == "__main__":
    unittest.main()
