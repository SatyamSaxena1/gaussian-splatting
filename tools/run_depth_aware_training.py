#!/usr/bin/env python3
"""Depth-aware Gaussian Splatting automation driver.

This utility stitches together the core steps required to integrate
Video-Depth-Anything (VDA) supervision into the Gaussian Splatting
training pipeline:

1. Ensure COLMAP data exists for the target scene (optionally rerun).
2. Align merged VDA depth frames with the COLMAP image set.
3. Regenerate per-image depth scale/offset using the COLMAP geometry.
4. Launch depth-regularized training followed by reference renders.

The script is intentionally opinionated for the pantograph project but
accepts CLI flags so it can be reused for future scenes once the VDA
assets are available.  Use ``--dry-run`` to preview the actions without
mutating the workspace.

Example:
	python tools/run_depth_aware_training.py \
		--scene data/pantograph_scene \
		--depth-source data/pantograph_scene/vda_merged_depth \
		--model-path output/pantograph_vda_depth \
		--iterations 30000

"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _resolve_path(root: Path, value: str) -> Path:
	"""Resolve a potentially relative path against the repository root."""

	path = Path(value)
	return path.resolve() if path.is_absolute() else (root / path).resolve()


def _extract_trailing_int(stem: str) -> Optional[int]:
	"""Return trailing integer from a filename stem (e.g. frame_00001 -> 1)."""

	match = re.search(r"(\d+)$", stem)
	return int(match.group(1)) if match else None


@dataclass
class DriverArgs:
	scene: str
	depth_source: Optional[str]
	depth_dir_name: str
	image_dir: str
	depth_index_offset: int
	skip_colmap: bool
	force_colmap: bool
	skip_depth_sync: bool
	force_depth_sync: bool
	skip_scale: bool
	skip_training: bool
	skip_render: bool
	dry_run: bool
	colmap_exe: str
	colmap_matching: str
	sequential_overlap: int
	sift_max_image_size: int
	sift_gpu_index: str
	sift_num_threads: int
	skip_matching: bool
	resize_images: bool
	gpu_env_script: str
	model_path: Optional[str]
	overwrite_model_path: bool
	iterations: int
	test_iterations: Sequence[int]
	save_iterations: Sequence[int]
	checkpoint_iterations: Sequence[int]
	train_extra: Sequence[str]
	data_device: Optional[str]
	render_iteration: Optional[int]


class DepthAwareTrainingDriver:
	"""Controller class orchestrating the end-to-end workflow."""

	def __init__(self, raw_args: argparse.Namespace):
		self.repo_root = Path(__file__).resolve().parents[1]
		args = DriverArgs(
			scene=raw_args.scene,
			depth_source=raw_args.depth_source,
			depth_dir_name=raw_args.depth_dir_name,
			image_dir=raw_args.image_dir,
			depth_index_offset=raw_args.depth_index_offset,
			skip_colmap=raw_args.skip_colmap,
			force_colmap=raw_args.force_colmap,
			skip_depth_sync=raw_args.skip_depth_sync,
			force_depth_sync=raw_args.force_depth_sync,
			skip_scale=raw_args.skip_scale,
			skip_training=raw_args.skip_training,
			skip_render=raw_args.skip_render,
			dry_run=raw_args.dry_run,
			colmap_exe=raw_args.colmap_exe,
			colmap_matching=raw_args.colmap_matching,
			sequential_overlap=raw_args.sequential_overlap,
			sift_max_image_size=raw_args.sift_max_image_size,
			sift_gpu_index=raw_args.sift_gpu_index,
			sift_num_threads=raw_args.sift_num_threads,
			skip_matching=raw_args.skip_matching,
			resize_images=raw_args.resize_images,
			gpu_env_script=raw_args.gpu_env_script,
			model_path=raw_args.model_path,
			overwrite_model_path=raw_args.overwrite_model_path,
			iterations=raw_args.iterations,
			test_iterations=raw_args.test_iterations,
			save_iterations=raw_args.save_iterations,
			checkpoint_iterations=raw_args.checkpoint_iterations,
			train_extra=raw_args.train_extra,
			data_device=raw_args.data_device,
			render_iteration=raw_args.render_iteration,
		)
		self.args = args

		self.scene_dir = _resolve_path(self.repo_root, args.scene)
		depth_source = args.depth_source or str(self.scene_dir / "vda_merged_depth")
		self.depth_source = _resolve_path(self.repo_root, depth_source)
		self.depth_target = (self.scene_dir / args.depth_dir_name).resolve()
		self.image_dir = (self.scene_dir / args.image_dir).resolve()
		self.colmap_exe = _resolve_path(self.repo_root, args.colmap_exe)
		self.gpu_env_script = _resolve_path(self.repo_root, args.gpu_env_script)

		if args.model_path:
			model_path = _resolve_path(self.repo_root, args.model_path)
		else:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			default_name = f"{self.scene_dir.name}_vda_depth_{timestamp}"
			model_path = (self.repo_root / "output" / default_name).resolve()
		self.model_path = model_path

		self.manifest: Dict[str, object] = {
			"scene": str(self.scene_dir),
			"depth_source": str(self.depth_source),
			"depth_target": str(self.depth_target),
			"image_dir": str(self.image_dir),
			"model_path": str(self.model_path),
			"args": vars(raw_args),
			"status": "pending",
			"steps": [],
			"started_at": datetime.utcnow().isoformat() + "Z",
		}
		self.manifest_path: Optional[Path] = None
		self._colmap_completed = False
		self._depth_sync_completed = False
		self._training_completed = False

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------
	def run(self) -> None:
		try:
			if self.args.skip_colmap:
				self._log_step("colmap", "skipped", "--skip-colmap provided")
			else:
				self._ensure_colmap_assets()

			if self.args.skip_depth_sync:
				self._log_step("depth_sync", "skipped", "--skip-depth-sync provided")
			else:
				self._sync_depth_frames()

			if self.args.skip_scale:
				self._log_step("depth_scale", "skipped", "--skip-scale provided")
			else:
				self._generate_depth_params()

			if self.args.skip_training:
				self._log_step("training", "skipped", "--skip-training provided")
			else:
				self._run_training()

			if self.args.skip_render:
				self._log_step("render", "skipped", "--skip-render provided")
			else:
				self._run_render()
			self.manifest["status"] = "success"
		except Exception as exc:  # pragma: no cover - defensive logging
			self.manifest["status"] = "failed"
			self.manifest["error"] = str(exc)
			raise
		finally:
			self._write_manifest()

	# ------------------------------------------------------------------
	# Individual steps
	# ------------------------------------------------------------------
	def _ensure_colmap_assets(self) -> None:
		step = "colmap"
		if not self.scene_dir.exists():
			raise FileNotFoundError(f"Scene directory not found: {self.scene_dir}")

		sparse_flag = self.scene_dir / "sparse" / "0" / "cameras.bin"
		needs_colmap = self.args.force_colmap or not sparse_flag.exists()

		if not needs_colmap:
			self._colmap_completed = True
			self._log_step(step, "skipped", "COLMAP artifacts already present")
			return

		cmd = [
			sys.executable,
			str((self.repo_root / "convert.py").resolve()),
			"-s",
			str(self.scene_dir),
			"--matching",
			self.args.colmap_matching,
			"--sequential_overlap",
			str(self.args.sequential_overlap),
		]
		if self.args.sift_max_image_size > 0:
			cmd.extend(["--sift_max_image_size", str(self.args.sift_max_image_size)])
		if self.args.sift_gpu_index:
			cmd.extend(["--sift_gpu_index", self.args.sift_gpu_index])
		if self.args.sift_num_threads > 0:
			cmd.extend(["--sift_num_threads", str(self.args.sift_num_threads)])
		if self.args.skip_matching:
			cmd.append("--skip_matching")
		if self.args.resize_images:
			cmd.append("--resize")
		cmd.extend(["--colmap_executable", str(self.colmap_exe)])

		executed = self._run_cmd(cmd, step)

		if not executed:
			self._colmap_completed = True
			self._log_step(step, "pending", "dry-run: convert.py command staged")
			return

		if not sparse_flag.exists():
			raise RuntimeError("COLMAP finished but sparse model missing (check logs)")
		if not self.image_dir.exists():
			raise RuntimeError(
				f"Expected undistorted images in {self.image_dir}, but the folder is missing."
			)
		self._colmap_completed = True
		self._log_step(step, "ok", "COLMAP conversion completed")

	def _sync_depth_frames(self) -> None:
		step = "depth_sync"
		if not self.depth_source.exists():
			raise FileNotFoundError(f"Depth source not found: {self.depth_source}")
		if not self.image_dir.exists():
			raise FileNotFoundError(
				f"Image directory not found (run COLMAP first): {self.image_dir}"
			)

		if self.depth_target.exists():
			if self.args.force_depth_sync:
				if not self.args.dry_run:
					shutil.rmtree(self.depth_target)
			else:
				self._depth_sync_completed = True
				self._log_step(step, "skipped", "Depth directory already exists")
				return

		if not self.args.dry_run:
			self.depth_target.mkdir(parents=True, exist_ok=True)

		depth_lookup = self._build_depth_lookup()
		image_files = self._collect_image_files()
		if not image_files:
			raise RuntimeError(f"No training images found under {self.image_dir}")
		if not depth_lookup:
			raise RuntimeError(f"No depth PNGs detected in {self.depth_source}")

		missing: List[str] = []
		copied = 0
		for img in image_files:
			frame_idx = _extract_trailing_int(img.stem)
			if frame_idx is None:
				missing.append(img.name)
				continue
			depth_idx = frame_idx - self.args.depth_index_offset
			depth_path = depth_lookup.get(depth_idx)
			if not depth_path:
				missing.append(img.name)
				continue
			target_path = self.depth_target / f"{img.stem}.png"
			if not self.args.dry_run:
				shutil.copy2(depth_path, target_path)
			copied += 1
		summary = f"Copied {copied} depth frames into {self.depth_target.name}"
		if missing:
			summary += f"; skipped {len(missing)} frames without matching depth"
		if self.args.dry_run:
			summary += " (dry-run: files not modified)"
		self._depth_sync_completed = True
		self._log_step(step, "ok", summary)

	def _generate_depth_params(self) -> None:
		step = "depth_scale"
		sparse_dir = self.scene_dir / "sparse" / "0"
		if not sparse_dir.exists() and not (self.args.dry_run and self._colmap_completed):
			raise RuntimeError("Cannot build depth scale without COLMAP sparse model")
		if not self.depth_target.exists() and not (self.args.dry_run and self._depth_sync_completed):
			raise RuntimeError("Depth directory missing; run depth sync first")

		cmd = [
			sys.executable,
			str((self.repo_root / "utils" / "make_depth_scale.py").resolve()),
			"--base_dir",
			str(self.scene_dir),
			"--depths_dir",
			str(self.depth_target),
		]
		executed = self._run_cmd(cmd, step)
		if executed:
			self._log_step(step, "ok", "depth_params.json regenerated")
		else:
			self._log_step(step, "pending", "dry-run: depth scale command staged")

	def _run_training(self) -> None:
		step = "training"
		if self.model_path.exists() and not self.args.overwrite_model_path:
			self._log_step(step, "skipped", "Model path already exists; use --overwrite-model-path")
			return
		if self.model_path.exists() and self.args.overwrite_model_path and not self.args.dry_run:
			shutil.rmtree(self.model_path)

		train_cmd = [
			str(self.gpu_env_script),
			sys.executable,
			str((self.repo_root / "train.py").resolve()),
			"-s",
			str(self.scene_dir),
			"-m",
			str(self.model_path),
			"-d",
			self.args.depth_dir_name,
			"--iterations",
			str(self.args.iterations),
		]
		if self.args.data_device:
			train_cmd.extend(["--data_device", self.args.data_device])
		for iteration_list, flag in (
			(self.args.test_iterations, "--test_iterations"),
			(self.args.save_iterations, "--save_iterations"),
			(self.args.checkpoint_iterations, "--checkpoint_iterations"),
		):
			if iteration_list:
				train_cmd.append(flag)
				train_cmd.extend(str(it) for it in iteration_list)
		if self.args.train_extra:
			train_cmd.extend(self.args.train_extra)

		executed = self._run_cmd(train_cmd, step)
		if executed:
			self._training_completed = True
			self._log_step(step, "ok", f"Training finished ({self.args.iterations} iters)")
		else:
			self._training_completed = True
			self._log_step(step, "pending", "dry-run: training command staged")

	def _run_render(self) -> None:
		step = "render"
		if not self.model_path.exists() and not (self.args.dry_run and self._training_completed):
			self._log_step(step, "skipped", "Model directory missing (training skipped?)")
			return
		iteration = self.args.render_iteration or self.args.iterations
		render_cmd = [
			str(self.gpu_env_script),
			sys.executable,
			str((self.repo_root / "render.py").resolve()),
			"--model_path",
			str(self.model_path),
			"--iteration",
			str(iteration),
		]
		executed = self._run_cmd(render_cmd, step)
		if executed:
			self._log_step(step, "ok", f"Render set completed @ iteration {iteration}")
		else:
			self._log_step(step, "pending", "dry-run: render command staged")

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------
	def _log_step(self, name: str, status: str, message: str) -> None:
		entry = {
			"name": name,
			"status": status,
			"message": message,
			"timestamp": datetime.utcnow().isoformat() + "Z",
		}
		self.manifest.setdefault("steps", []).append(entry)
		print(f"[{name}] {status.upper()}: {message}")

	def _run_cmd(self, cmd: Sequence[str], step_name: str) -> bool:
		cmd_str = " ".join(str(part) for part in cmd)
		print(f"[{step_name}] $ {cmd_str}")
		if self.args.dry_run:
			print(f"[{step_name}] dry-run mode: command not executed")
			return False
		subprocess.run(cmd, check=True, cwd=self.repo_root)
		return True

	def _collect_image_files(self) -> List[Path]:
		files: List[Path] = []
		for ext in ("*.jpg", "*.jpeg", "*.png"):
			files.extend(sorted(self.image_dir.glob(ext)))
		return sorted(files, key=lambda path: (_extract_trailing_int(path.stem) or -1, path.name))

	def _build_depth_lookup(self) -> Dict[int, Path]:
		lookup: Dict[int, Path] = {}
		for path in sorted(self.depth_source.glob("depth_*.png")):
			idx = _extract_trailing_int(path.stem)
			if idx is None:
				continue
			lookup[idx] = path
		return lookup

	def _write_manifest(self) -> None:
		runs_dir = self.scene_dir / "logs" / "depth_driver_runs"
		if not runs_dir.exists() and not self.args.dry_run:
			runs_dir.mkdir(parents=True, exist_ok=True)
		timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
		manifest_path = runs_dir / f"run_{timestamp}.json"
		self.manifest_path = manifest_path
		if self.args.dry_run:
			print(f"[manifest] dry-run mode: manifest not written (would be {manifest_path})")
			return
		with manifest_path.open("w", encoding="utf-8") as fp:
			json.dump(self.manifest, fp, indent=2)
		print(f"[manifest] saved run metadata to {manifest_path}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Depth-aware Gaussian Splatting driver")
	parser.add_argument("--scene", default="data/pantograph_scene", help="Scene directory passed to train.py (-s)")
	parser.add_argument("--depth-source", help="Directory containing merged VDA depth PNGs")
	parser.add_argument("--depth-dir-name", default="depths_vda", help="Relative directory name (under scene) for aligned depths")
	parser.add_argument("--image-dir", default="images", help="Relative image directory to align against (usually 'images')")
	parser.add_argument("--depth-index-offset", type=int, default=1, help="Offset between frame index and depth index (frame_idx - offset -> depth idx)")

	parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP verification / conversion step")
	parser.add_argument("--force-colmap", action="store_true", help="Force rerun of convert.py even if sparse model exists")
	parser.add_argument("--skip-depth-sync", action="store_true")
	parser.add_argument("--force-depth-sync", action="store_true")
	parser.add_argument("--skip-scale", action="store_true", help="Skip make_depth_scale execution")
	parser.add_argument("--skip-training", action="store_true")
	parser.add_argument("--skip-render", action="store_true")
	parser.add_argument("--dry-run", action="store_true", help="Print planned commands without changing files")

	parser.add_argument("--colmap-exe", default="./run_colmap_local.sh", help="Wrapper used for convert.py --colmap_executable")
	parser.add_argument("--colmap-matching", default="sequential", choices=("sequential", "exhaustive"))
	parser.add_argument("--sequential-overlap", type=int, default=5)
	parser.add_argument("--sift-max-image-size", type=int, default=1600)
	parser.add_argument("--sift-gpu-index", default="0")
	parser.add_argument("--sift-num-threads", type=int, default=8)
	parser.add_argument("--skip-matching", action="store_true", help="Forwarded to convert.py")
	parser.add_argument("--resize-images", action="store_true", help="Forward convert.py --resize")

	parser.add_argument("--gpu-env-script", default="./gpu_env.sh", help="Wrapper script to launch GPU-enabled Python")
	parser.add_argument("--model-path", help="Explicit output directory for training results")
	parser.add_argument("--overwrite-model-path", action="store_true", help="Delete existing model directory before training")
	parser.add_argument("--iterations", type=int, default=30000)
	parser.add_argument("--test-iterations", nargs="*", default=[7000, 30000], type=int)
	parser.add_argument("--save-iterations", nargs="*", default=[7000, 30000], type=int)
	parser.add_argument("--checkpoint-iterations", nargs="*", default=[], type=int)
	parser.add_argument("--data-device", help="Forwarded to train.py --data_device")
	parser.add_argument("--render-iteration", type=int, help="Iteration to render (defaults to --iterations)")
	parser.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[], help="Additional flags appended to the train.py command")
	return parser


def main() -> None:
	parser = build_arg_parser()
	raw_args = parser.parse_args()
	driver = DepthAwareTrainingDriver(raw_args)
	driver.run()


if __name__ == "__main__":
	main()

