#!/usr/bin/env python3
"""Enhanced USD exporter for Gaussian Splatting checkpoints."""
from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

try:
    from plyfile import PlyData
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("The 'plyfile' package is required to run export_usd.py") from exc

C0 = 0.28209479177387814  # Spherical Harmonics constant for DC term


@dataclass
class GaussianData:
    positions: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    features_rest: Optional[np.ndarray]
    sh_degree: int

    def subset(self, mask: np.ndarray) -> "GaussianData":
        return GaussianData(
            positions=self.positions[mask],
            colors=self.colors[mask],
            opacities=self.opacities[mask],
            scales=self.scales[mask],
            rotations=self.rotations[mask],
            features_rest=None if self.features_rest is None else self.features_rest[mask],
            sh_degree=self.sh_degree,
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return quat / norm


def _read_gaussians(path: Path) -> GaussianData:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]

    positions = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=-1).astype(np.float32)

    dc = np.stack((vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]), axis=-1).astype(np.float32)
    colors = np.clip(dc * C0 + 0.5, 0.0, 1.0)

    opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))

    scale_names = sorted(name for name in vertex.data.dtype.names if name.startswith("scale_"))
    scales = np.stack([vertex[name] for name in scale_names], axis=-1).astype(np.float32)
    scales = np.exp(scales)

    rot_names = sorted(name for name in vertex.data.dtype.names if name.startswith("rot_"))
    rotations = np.stack([vertex[name] for name in rot_names], axis=-1).astype(np.float32)
    rotations = _normalize_quaternion(rotations)

    rest_names = sorted(name for name in vertex.data.dtype.names if name.startswith("f_rest_"))
    features_rest: Optional[np.ndarray]
    if rest_names:
        features_rest = np.stack([vertex[name] for name in rest_names], axis=-1).astype(np.float32)
        coeffs_per_channel = features_rest.shape[-1] + 1
        sh_degree = int(math.isqrt(coeffs_per_channel) - 1)
    else:
        features_rest = None
        sh_degree = 0

    return GaussianData(
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        features_rest=features_rest,
        sh_degree=sh_degree,
    )


def _maybe_downsample(data: GaussianData, max_points: Optional[int], every_n: Optional[int]) -> GaussianData:
    count = data.positions.shape[0]
    if count == 0:
        return data

    if every_n and every_n > 1:
        mask = np.arange(count) % every_n == 0
        return data.subset(mask)

    if max_points and max_points < count:
        rng = np.random.default_rng()
        indices = rng.choice(count, size=max_points, replace=False)
        return data.subset(indices)

    return data


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rot = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    rot[:, 0, 0] = ww + xx - yy - zz
    rot[:, 0, 1] = 2 * (xy - wz)
    rot[:, 0, 2] = 2 * (xz + wy)
    rot[:, 1, 0] = 2 * (xy + wz)
    rot[:, 1, 1] = ww - xx + yy - zz
    rot[:, 1, 2] = 2 * (yz - wx)
    rot[:, 2, 0] = 2 * (xz - wy)
    rot[:, 2, 1] = 2 * (yz + wx)
    rot[:, 2, 2] = ww - xx - yy + zz
    return rot


def _extent_from_splats(data: GaussianData) -> tuple[np.ndarray, np.ndarray]:
    max_scale = np.max(data.scales, axis=1, keepdims=True)
    minimum = np.min(data.positions - max_scale, axis=0)
    maximum = np.max(data.positions + max_scale, axis=0)
    return minimum, maximum


def _build_point_instancer(stage: Usd.Stage, data: GaussianData, proto_radius: float) -> UsdGeom.PointInstancer:
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    proto_root = UsdGeom.Xform.Define(stage, "/World/Prototypes")
    proto_sphere = UsdGeom.Sphere.Define(stage, "/World/Prototypes/Gaussian")
    proto_sphere.CreateRadiusAttr(max(float(proto_radius), 1e-6))

    instancer = UsdGeom.PointInstancer.Define(stage, "/World/Cloud")
    instancer.GetPrototypesRel().AddTarget(proto_sphere.GetPath())

    count = data.positions.shape[0]

    proto_indices = Vt.IntArray(count)
    for idx in range(count):
        proto_indices[idx] = 0
    instancer.CreateProtoIndicesAttr(proto_indices)

    positions = Vt.Vec3fArray(count)
    for idx, point in enumerate(data.positions):
        positions[idx] = Gf.Vec3f(*map(float, point))
    instancer.CreatePositionsAttr(positions)

    scales = Vt.Vec3fArray(count)
    for idx, scale in enumerate(data.scales):
        scales[idx] = Gf.Vec3f(*map(float, scale))
    instancer.CreateScalesAttr(scales)

    orientations = Vt.QuathArray(count)
    for idx, quat in enumerate(data.rotations):
        orientations[idx] = Gf.Quath(float(quat[0]), Gf.Vec3h(*map(float, quat[1:4])))
    instancer.CreateOrientationsAttr(orientations)

    # Appearance primvars
    primvars = UsdGeom.PrimvarsAPI(instancer)

    colors = Vt.Vec3fArray(count)
    for idx, color in enumerate(data.colors):
        colors[idx] = Gf.Vec3f(*map(float, color))
    primvars.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(colors)

    opacity = Vt.FloatArray(count)
    for idx, value in enumerate(data.opacities):
        opacity[idx] = float(value)
    primvars.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(opacity)

    # Useful analytics
    rotation_mats = _quat_to_matrix(data.rotations)
    principal_axis = rotation_mats[:, :, 2]

    principal_prim = Vt.Vec3fArray(count)
    for idx, axis in enumerate(principal_axis):
        principal_prim[idx] = Gf.Vec3f(*map(float, axis))
    primvars.CreatePrimvar("gaussianPrincipalAxis", Sdf.ValueTypeNames.Vector3fArray, UsdGeom.Tokens.vertex).Set(principal_prim)

    scale_prim = Vt.Vec3fArray(count)
    for idx, scale in enumerate(data.scales):
        scale_prim[idx] = Gf.Vec3f(*map(float, scale))
    primvars.CreatePrimvar("gaussianScale", Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex).Set(scale_prim)

    volume = np.prod(data.scales, axis=1)
    volume_prim = Vt.FloatArray(count)
    for idx, value in enumerate(volume):
        volume_prim[idx] = float(value)
    primvars.CreatePrimvar("gaussianVolume", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(volume_prim)

    ecc = np.max(data.scales, axis=1) / np.maximum(np.min(data.scales, axis=1), 1e-6)
    ecc_prim = Vt.FloatArray(count)
    for idx, value in enumerate(ecc):
        ecc_prim[idx] = float(value)
    primvars.CreatePrimvar("gaussianEccentricity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(ecc_prim)

    degree = Vt.IntArray(count)
    for idx in range(count):
        degree[idx] = int(data.sh_degree)
    primvars.CreatePrimvar("shDegree", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(degree)

    mn, mx = _extent_from_splats(data)
    instancer.CreateExtentAttr([Gf.Vec3f(*map(float, mn)), Gf.Vec3f(*map(float, mx))])

    return instancer


def _to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _add_detections(stage: Usd.Stage, detections_path: Path) -> int:
    with detections_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload if isinstance(payload, list) else [payload]

    scope = UsdGeom.Xform.Define(stage, "/World/Detections")
    added = 0
    for index, item in enumerate(items):
        prim_path = f"{scope.GetPath()}/Detection_{index:03d}"
        start = item.get("bottom_endpoint")
        end = item.get("top_endpoint")
        radius = item.get("radius", 0.05)
        if start is None or end is None:
            continue
        cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
        start_vec = Gf.Vec3d(*map(float, start))
        end_vec = Gf.Vec3d(*map(float, end))
        length_vec = end_vec - start_vec
        length = length_vec.GetLength()
        if length == 0.0:
            continue
        dir_vec = Gf.Vec3d(length_vec)
        dir_vec.Normalize()

        cylinder.CreateRadiusAttr(float(radius))
        cylinder.CreateHeightAttr(float(length))

        xformable = UsdGeom.Xformable(cylinder.GetPrim())
        rotation = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), dir_vec)
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(rotation)
        matrix.SetTranslate((start_vec + end_vec) * 0.5)
        xformable.AddTransformOp().Set(matrix)

        prim = cylinder.GetPrim()
        for key, value in item.items():
            prim.SetCustomDataByKey(key, _to_serializable(value))
        added += 1

    return added


def _parse_namespace_config(cfg_path: Path) -> Dict[str, object]:
    try:
        text = cfg_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return {}

    if not text.startswith("Namespace(") or not text.endswith(")"):
        return {"raw": text}

    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError:
        return {"raw": text}

    if not isinstance(node.body, ast.Call) or getattr(node.body.func, "id", None) != "Namespace":
        return {"raw": text}

    result: Dict[str, object] = {}
    for keyword in node.body.keywords:
        try:
            result[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            result[keyword.arg] = "<unparsed>"
    return result


def _load_exposures(model_path: Path) -> Optional[Dict[str, object]]:
    exposure_path = model_path / "exposure.json"
    if not exposure_path.exists():
        return None
    try:
        with exposure_path.open("r", encoding="utf-8") as handle:
            exposures = json.load(handle)
    except json.JSONDecodeError:
        return None

    summary = {
        "count": len(exposures),
    }
    return {"summary": summary}


def _load_cameras(model_path: Path) -> Optional[list[dict]]:
    cam_path = model_path / "cameras.json"
    if not cam_path.exists():
        return None
    with cam_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _add_cameras(stage: Usd.Stage, cameras: list[dict]) -> None:
    root = UsdGeom.Xform.Define(stage, "/World/Cameras")
    for cam in cameras:
        cam_path = f"/World/Cameras/Camera_{cam['id']:04d}"
        camera = UsdGeom.Camera.Define(stage, cam_path)

        rotation_w2c = np.array(cam["rotation"], dtype=np.float64)
        translation_w2c = np.array(cam["position"], dtype=np.float64)
        rotation_c2w = rotation_w2c.T
        camera_center = -rotation_c2w @ translation_w2c

        matrix = Gf.Matrix4d(1.0)
        for r in range(3):
            for c in range(3):
                matrix[r][c] = float(rotation_c2w[r, c])
        matrix[0][3] = float(camera_center[0])
        matrix[1][3] = float(camera_center[1])
        matrix[2][3] = float(camera_center[2])
        matrix[3][0] = matrix[3][1] = matrix[3][2] = 0.0
        matrix[3][3] = 1.0

        UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(matrix)

        width = float(cam.get("width", 1920))
        height = float(cam.get("height", 1080))
        fx = float(cam.get("fx", 1000.0))
        fy = float(cam.get("fy", 1000.0))

        camera.CreateHorizontalApertureAttr(width)
        camera.CreateVerticalApertureAttr(height)
        camera.CreateFocalLengthAttr((fx + fy) * 0.5)
        camera.CreateFocusDistanceAttr(1.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

        prim = camera.GetPrim()
        prim.SetCustomDataByKey("img_name", cam.get("img_name"))
        prim.SetCustomDataByKey("fx", fx)
        prim.SetCustomDataByKey("fy", fy)


def _collect_metadata(cloud_path: Path, data: GaussianData, cfg: Dict[str, object]) -> Dict[str, object]:
    iteration_dir = cloud_path.parent.name
    try:
        iteration = int(iteration_dir.split("_")[-1])
    except ValueError:
        iteration = None

    stats = {
        "count": int(data.positions.shape[0]),
        "maxScale": float(np.max(data.scales)),
        "minScale": float(np.min(data.scales)),
        "meanScale": float(np.mean(data.scales)),
        "maxOpacity": float(np.max(data.opacities)),
        "minOpacity": float(np.min(data.opacities)),
        "meanOpacity": float(np.mean(data.opacities)),
        "shDegree": int(data.sh_degree),
        "iteration": iteration,
        "cloudPath": str(cloud_path),
    }

    metadata = {
        "exporter": {
            "tool": "tools/export_usd.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "gaussians": stats,
    }
    if cfg:
        metadata["config"] = cfg
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gaussian splats, metadata, and overlays to USD")
    parser.add_argument("--cloud", required=True, type=Path, help="Path to point_cloud.ply")
    parser.add_argument("--det", type=Path, default=None, help="Optional detection JSON to visualize")
    parser.add_argument("--out", type=Path, default=Path("output/run.usda"), help="Destination USD stage")
    parser.add_argument("--meters", type=float, default=1.0, help="Meters-per-unit metadata for the stage")
    parser.add_argument("--stage-up", choices=["y", "z"], default="y", help="Stage up-axis")
    parser.add_argument("--max-points", type=int, default=None, help="Randomly sample at most N splats")
    parser.add_argument("--every-n", type=int, default=None, help="Keep every Nth splat deterministically")
    parser.add_argument("--ptsize", type=float, default=0.01, help="Compatibility flag (prototype radius multiplier)")
    args = parser.parse_args()

    cloud_path = args.cloud.resolve()
    if not cloud_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {cloud_path}")

    data = _read_gaussians(cloud_path)
    data = _maybe_downsample(data, args.max_points, args.every_n)

    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    stage_up_token = UsdGeom.Tokens.y if args.stage_up == "y" else UsdGeom.Tokens.z
    UsdGeom.SetStageUpAxis(stage, stage_up_token)
    UsdGeom.SetStageMetersPerUnit(stage, float(args.meters))

    instancer = _build_point_instancer(stage, data, args.ptsize)
    instancer.GetPrim().SetCustomDataByKey("pointCount", int(data.positions.shape[0]))

    model_path = cloud_path.parent.parent.parent
    cfg = _parse_namespace_config(model_path / "cfg_args")
    exposures = _load_exposures(model_path)
    cameras = _load_cameras(model_path)

    detection_summary = None
    if args.det:
        det_path = args.det.resolve()
        if det_path.exists():
            added = _add_detections(stage, det_path)
            detection_summary = {"path": str(det_path), "count": int(added)}
        else:
            print(f"[export_usd] Detection file not found: {det_path}; skipping overlays")

    if cameras:
        _add_cameras(stage, cameras)
        camera_count = len(cameras)
    else:
        camera_count = 0

    layer = stage.GetRootLayer()
    metadata = _collect_metadata(cloud_path, data, cfg)
    if exposures:
        metadata["exposures"] = exposures
    if detection_summary:
        metadata["detections"] = detection_summary
    metadata["cameras"] = {"count": camera_count}
    layer.customLayerData = metadata

    stage.GetRootLayer().Save()
    print(f"[export_usd] Wrote stage to {output_path}")


if __name__ == "__main__":
    main()
