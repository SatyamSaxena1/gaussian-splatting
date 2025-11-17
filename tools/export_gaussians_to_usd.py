#!/usr/bin/env python3
"""Export trained Gaussian splats to a USD stage for visualization in usdview."""
import argparse
from pathlib import Path
from typing import Optional

import importlib

import numpy as np
from plyfile import PlyData

try:
    pxr = importlib.import_module("pxr")
except ImportError as exc:
    raise ImportError(
        "Could not import 'pxr'. Activate the USD Python environment (see docs/USDVIEW_SETUP.md)."
    ) from exc

Gf = pxr.Gf
Sdf = pxr.Sdf
Usd = pxr.Usd
UsdGeom = pxr.UsdGeom
Vt = pxr.Vt

from utils.system_utils import searchForMaxIteration


C0 = 0.28209479177387814  # Spherical harmonics constant for DC term


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sh_to_rgb(dc_coeffs: np.ndarray) -> np.ndarray:
    """Approximate RGB from the DC spherical harmonics coefficients."""
    rgb = dc_coeffs * C0 + 0.5
    return np.clip(rgb, 0.0, 1.0)


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return quat / norm


def _resolve_point_cloud(model_path: Path, iteration: Optional[int]) -> Path:
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.exists():
        raise FileNotFoundError(f"Could not find point_cloud directory under {model_path}")

    if iteration is None:
        iteration = searchForMaxIteration(str(point_cloud_root))

    ply_path = point_cloud_root / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Could not find point_cloud.ply at iteration {iteration} in {model_path}")

    return ply_path


def _read_gaussians(ply_path: Path) -> dict:
    ply = PlyData.read(str(ply_path))
    vertex = ply.elements[0]

    positions = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=-1).astype(np.float32)
    dc = np.stack((vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]), axis=-1).astype(np.float32)
    colors = _sh_to_rgb(dc)

    opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))

    scale_columns = [name for name in vertex.data.dtype.names if name.startswith("scale_")]
    scale_columns = sorted(scale_columns, key=lambda n: int(n.split("_")[-1]))
    scales = np.stack([vertex[name] for name in scale_columns], axis=-1).astype(np.float32)
    scales = np.exp(scales)  # stored as log-scale in the checkpoints

    rot_columns = [name for name in vertex.data.dtype.names if name.startswith("rot_")]
    rot_columns = sorted(rot_columns, key=lambda n: int(n.split("_")[-1]))
    rotations = np.stack([vertex[name] for name in rot_columns], axis=-1).astype(np.float32)
    rotations = _normalize_quaternion(rotations)

    return {
        "positions": positions,
        "colors": colors,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
    }


def _maybe_downsample(attributes: dict, max_points: Optional[int], every_n: Optional[int]) -> dict:
    count = attributes["positions"].shape[0]
    if count == 0:
        return attributes

    if every_n and every_n > 1:
        mask = np.arange(count) % every_n == 0
    elif max_points and max_points < count:
        rng = np.random.default_rng()
        indices = rng.choice(count, size=max_points, replace=False)
        mask = np.zeros(count, dtype=bool)
        mask[indices] = True
    else:
        return attributes

    return {key: value[mask] for key, value in attributes.items()}


def _build_stage(output_path: Path, data: dict, stage_up: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if stage_up.lower() == "y":
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    elif stage_up.lower() == "z":
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    else:
        raise ValueError("stage_up must be either 'y' or 'z'")

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    proto_root = UsdGeom.Xform.Define(stage, "/World/Prototypes")
    proto_sphere = UsdGeom.Sphere.Define(stage, "/World/Prototypes/Gaussian")
    proto_sphere.CreateRadiusAttr(1.0)

    instancer = UsdGeom.PointInstancer.Define(stage, "/World/GaussianSplats")
    instancer.GetPrototypesRel().AddTarget(proto_sphere.GetPath())

    num = data["positions"].shape[0]

    proto_indices = Vt.IntArray(num)
    for i in range(num):
        proto_indices[i] = 0
    instancer.CreateProtoIndicesAttr(proto_indices)

    positions = Vt.Vec3fArray(num)
    scales = Vt.Vec3fArray(num)
    orientations = Vt.QuatfArray(num)
    colors = Vt.Vec3fArray(num)
    opacities = Vt.FloatArray(num)

    for idx in range(num):
        px, py, pz = data["positions"][idx]
        sx, sy, sz = data["scales"][idx]
        w, x, y, z = data["rotations"][idx]
        cr, cg, cb = data["colors"][idx]

        positions[idx] = Gf.Vec3f(px, py, pz)
        scales[idx] = Gf.Vec3f(sx, sy, sz)
        orientations[idx] = Gf.Quatf(w, Gf.Vec3f(x, y, z))
        colors[idx] = Gf.Vec3f(cr, cg, cb)
        opacities[idx] = float(data["opacities"][idx])

    instancer.CreatePositionsAttr(positions)
    instancer.CreateScalesAttr(scales)
    instancer.CreateOrientationsAttr(orientations)

    color_primvar = instancer.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    color_primvar.Set(colors)

    opacity_primvar = instancer.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex)
    opacity_primvar.Set(opacities)

    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a trained Gaussian Splat model to USD for usdview.")
    parser.add_argument("--model-path", required=True, type=Path, help="Output directory produced by training (contains point_cloud/...)")
    parser.add_argument("--iteration", type=str, default="latest", help="Checkpoint iteration to export or 'latest'.")
    parser.add_argument("--stage-path", type=Path, default=None, help="Destination USD stage path. Defaults to <model-path>/gaussians.usda")
    parser.add_argument("--max-points", type=int, default=None, help="Optional limit on number of splats to export (random subset).")
    parser.add_argument("--every-n", type=int, default=None, help="Keep every Nth splat instead of random sampling.")
    parser.add_argument("--stage-up", type=str, default="y", choices=["y", "z"], help="Up axis for the generated stage.")

    args = parser.parse_args()

    iteration: Optional[int]
    if args.iteration.lower() == "latest":
        iteration = None
    else:
        try:
            iteration = int(args.iteration)
        except ValueError as exc:
            raise ValueError("iteration must be an integer or 'latest'") from exc

    ply_path = _resolve_point_cloud(args.model_path.resolve(), iteration)
    data = _read_gaussians(ply_path)
    data = _maybe_downsample(data, args.max_points, args.every_n)

    if args.stage_path is None:
        stage_path = args.model_path / "gaussians.usda"
    else:
        stage_path = args.stage_path

    _build_stage(stage_path.resolve(), data, args.stage_up)
    print(f"Exported {data['positions'].shape[0]} splats to {stage_path}")


if __name__ == "__main__":
    main()
