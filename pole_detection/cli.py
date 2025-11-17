"""Command-line interface for pole detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .detect import (
    detect_multiple_poles_from_file,
    detect_pole_from_file,
    detection_to_dict,
    detections_to_dicts,
    save_detection_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect pole-like cylinders in a point cloud")
    parser.add_argument("point_cloud", type=Path, help="Path to the input point cloud (.ply)")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the detection result as JSON",
    )
    parser.add_argument("--num-iterations", type=int, default=512, help="Number of RANSAC iterations")
    parser.add_argument("--sample-size", type=int, default=1024, help="Number of points sampled per iteration")
    parser.add_argument(
        "--radial-threshold",
        type=float,
        default=0.05,
        help="Tolerance when classifying radial inliers (in scene units)",
    )
    parser.add_argument("--min-inliers", type=int, default=800, help="Minimum inliers required for acceptance")
    parser.add_argument(
        "--min-height",
        type=float,
        default=1.0,
        help="Minimum height a cylinder must span to be valid",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Randomly subsample to this many points before fitting (0 disables)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for subsampling and RANSAC",
    )
    parser.add_argument(
        "--max-poles",
        type=int,
        default=1,
        help="Detect up to this many poles (0 means no limit)",
    )
    parser.add_argument(
        "--removal-padding",
        type=float,
        default=0.02,
        help="Additional margin (scene units) applied when removing detected pole points",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    point_cloud_path: Path = args.point_cloud
    if not point_cloud_path.exists():
        parser.error(f"Point cloud file '{point_cloud_path}' does not exist")

    max_points = None if args.max_points in (None, 0) else int(args.max_points)

    max_poles = args.max_poles if args.max_poles and args.max_poles > 0 else None

    if max_poles == 1:
        result = detect_pole_from_file(
            point_cloud_path,
            num_iterations=args.num_iterations,
            sample_size=args.sample_size,
            radial_threshold=args.radial_threshold,
            min_inliers=args.min_inliers,
            min_height=args.min_height,
            max_points=max_points,
            random_state=args.seed,
        )

        if result is None:
            print("No pole detected", file=sys.stderr)
            return 1

        payload = detection_to_dict(result)
        if args.output_json is not None:
            save_detection_json(args.output_json, result)

        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    results = detect_multiple_poles_from_file(
        point_cloud_path,
        num_iterations=args.num_iterations,
        sample_size=args.sample_size,
        radial_threshold=args.radial_threshold,
        min_inliers=args.min_inliers,
        min_height=args.min_height,
        max_points=max_points,
        max_poles=max_poles,
        removal_padding=args.removal_padding,
        random_state=args.seed,
    )

    if not results:
        print("No poles detected", file=sys.stderr)
        return 1

    payload = detections_to_dicts(results)
    if args.output_json is not None:
        save_detection_json(args.output_json, results)

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
