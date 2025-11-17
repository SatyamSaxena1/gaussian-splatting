"""Tests for the pole detection CLI."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np

from pole_detection import cli
from pole_detection.synthetic import (
    CylinderSpec,
    generate_cylinder_point_cloud,
    save_point_cloud_ply,
)


class TestPoleDetectionCLI(unittest.TestCase):
    def test_cli_detects_pole_and_writes_json(self) -> None:
        spec = CylinderSpec(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            radius=0.2,
            height=3.0,
        )
        points = generate_cylinder_point_cloud(
            spec,
            num_points=5000,
            noise_std=0.01,
            outlier_ratio=0.05,
            random_state=7,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cloud_path = tmp_path / "cloud.ply"
            json_path = tmp_path / "result.json"
            save_point_cloud_ply(cloud_path, points)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = cli.main(
                    [
                        str(cloud_path),
                        "--output-json",
                        str(json_path),
                        "--num-iterations",
                        "400",
                        "--sample-size",
                        "800",
                        "--radial-threshold",
                        "0.04",
                        "--min-inliers",
                        "1000",
                        "--min-height",
                        "2.5",
                        "--max-points",
                        "4000",
                        "--seed",
                        "123",
                    ]
                )

            self.assertEqual(exit_code, 0)
            stdout_payload = json.loads(buffer.getvalue())
            file_payload = json.loads(json_path.read_text(encoding="utf8"))

            self.assertAlmostEqual(stdout_payload["radius"], spec.radius, delta=0.05)
            self.assertAlmostEqual(file_payload["radius"], spec.radius, delta=0.05)
            self.assertLess(stdout_payload["lean_angle_degrees"], 5.0)

    def test_cli_returns_error_on_missing_file(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(["missing-file.ply"])

    def test_cli_detects_multiple_poles(self) -> None:
        spec_a = CylinderSpec(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            axis_direction=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            radius=0.18,
            height=2.6,
        )
        spec_b = CylinderSpec(
            center=np.array([2.5, 0.0, 0.5], dtype=np.float32),
            axis_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            radius=0.2,
            height=2.2,
        )

        points = np.vstack(
            [
                generate_cylinder_point_cloud(
                    spec_a,
                    num_points=3500,
                    noise_std=0.01,
                    outlier_ratio=0.02,
                    random_state=4,
                ),
                generate_cylinder_point_cloud(
                    spec_b,
                    num_points=3500,
                    noise_std=0.01,
                    outlier_ratio=0.02,
                    random_state=9,
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cloud_path = tmp_path / "multi.ply"
            json_path = tmp_path / "multi.json"
            save_point_cloud_ply(cloud_path, points)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = cli.main(
                    [
                        str(cloud_path),
                        "--output-json",
                        str(json_path),
                        "--num-iterations",
                        "600",
                        "--sample-size",
                        "512",
                        "--radial-threshold",
                        "0.05",
                        "--min-inliers",
                        "700",
                        "--min-height",
                        "1.6",
                        "--max-points",
                        "0",
                        "--max-poles",
                        "2",
                        "--removal-padding",
                        "0.05",
                        "--seed",
                        "21",
                    ]
                )

            self.assertEqual(exit_code, 0)
            stdout_payload = json.loads(buffer.getvalue())
            file_payload = json.loads(json_path.read_text(encoding="utf8"))

            self.assertIsInstance(stdout_payload, list)
            self.assertEqual(len(stdout_payload), 2)
            self.assertEqual(len(file_payload), 2)
            radii = sorted(item["radius"] for item in stdout_payload)
            self.assertAlmostEqual(radii[0], spec_a.radius, delta=0.08)
            self.assertAlmostEqual(radii[1], spec_b.radius, delta=0.08)


if __name__ == "__main__":
    unittest.main()
