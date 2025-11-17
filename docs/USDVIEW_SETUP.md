# USDView Integration

These steps wire usdview into the Gaussian Splatting workflow so you can inspect checkpoints while you iterate.

## Prerequisites
- Follow NVIDIA's usdview installation guide: https://docs.nvidia.com/learn-openusd/latest/usdview-install-instructions.html
- Extract the USD bundle to `~/Downloads/usd_root` (or set `USD_ROOT` to the extraction path).
- Install the missing X11 dependencies listed in the NVIDIA guide if you are on Ubuntu.

## Set Up the USD Python Environment
The prebuilt USD bundle ships with a matching Python runtime. Create a virtual environment once and reuse it:

```bash
cd "${USD_ROOT:-$HOME/Downloads/usd_root}"
./python/python -m venv ./python-usd-venv
source ./python-usd-venv/bin/activate
pip install --upgrade pip
pip install usd-core assimp-py==1.0.8
```

Whenever you want to export a USD stage, activate the environment:

```bash
source "${USD_ROOT:-$HOME/Downloads/usd_root}/python-usd-venv/bin/activate"
```

## Export a Checkpoint to USD
Run the exporter against an existing training output directory (the folder that contains `point_cloud/`):

```bash
python tools/export_gaussians_to_usd.py --model-path output/<run-name> --iteration latest --stage-path output/<run-name>/gaussians.usda
```

Useful flags:
- `--max-points <N>`: random subset if the scene is too dense.
- `--every-n <step>`: deterministic thinning by keeping every Nth splat.
- `--stage-up y|z`: choose the usdview up-axis (`y` matches the training viewer).

The exporter writes a `PointInstancer` stage with per-splat color, opacity, scale, and orientation data so you can orbit, filter, or slice inside usdview.

## Launch usdview
Use the helper to start usdview with the correct bundle:

```bash
chmod +x tools/open_usdview.sh  # once
USD_ROOT=~/Downloads/usd_root tools/open_usdview.sh output/<run-name>/gaussians.usda
```

The script falls back to `~/Downloads/usd_root` when `USD_ROOT` is not defined.

## Tips
- Large checkpoints can be heavy in usdview; prefer `--max-points` for interactive iteration.
- Regenerate the USD stage after each training run; the exporter always reads directly from the saved PLY checkpoint.
- You can add exported stages to your `.gitignore` or store them under `output/`, which is already ignored.
