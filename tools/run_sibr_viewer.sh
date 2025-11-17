#!/bin/bash
set -euo pipefail

usage() {
	cat <<'EOF'
Run the compiled SIBR Gaussian viewer on a trained model.

Usage: tools/run_sibr_viewer.sh --model output/<run_id> [--data data/<scene>] [--size width height] [--device gpu_id]

Environment variables (alternative to flags):
  MODEL_PATH=...     Same as --model
  DATA_PATH=...      Optional override for COLMAP data (defaults to <model>/source_path if available)
  RENDER_WIDTH=..., RENDER_HEIGHT=...
  CUDA_DEVICE=...    Viewer GPU index (default 0)

Examples:
  tools/run_sibr_viewer.sh --model output/e7c48eaa-9 --data data/whatsapp_20251030
  RENDER_WIDTH=1920 RENDER_HEIGHT=1080 tools/run_sibr_viewer.sh --model output/e7c48eaa-9
EOF
}

MODEL="${MODEL_PATH:-}"
DATA="${DATA_PATH:-}"
WIDTH="${RENDER_WIDTH:-}"
HEIGHT="${RENDER_HEIGHT:-}"
DEVICE="${CUDA_DEVICE:-0}"

while (( "$#" )); do
	case "$1" in
		--model) MODEL=$2; shift 2;;
		--data) DATA=$2; shift 2;;
		--size)
			WIDTH=$2; HEIGHT=$3; shift 3;;
		--device) DEVICE=$2; shift 2;;
		-h|--help) usage; exit 0;;
		*) echo "Unknown argument: $1"; usage; exit 1;;
	esac
done

if [ -z "$MODEL" ]; then
	echo "ERROR: model path required."
	usage
	exit 1
fi

if [ ! -d "$MODEL" ]; then
	echo "ERROR: model directory not found: $MODEL"
	exit 1
fi

VIEWER_BIN="$(dirname "$0")/../SIBR_viewers/install/bin/SIBR_gaussianViewer_app"
VIEWER_BIN="$(readlink -f "$VIEWER_BIN")"

if [ ! -x "$VIEWER_BIN" ]; then
	echo "ERROR: SIBR viewer binary not found or not executable: $VIEWER_BIN"
	echo "Build it with:"
	echo "  cd SIBR_viewers && cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc) --target install"
	exit 1
fi

CMD=( "$VIEWER_BIN" -m "$(readlink -f "$MODEL")" --device "$DEVICE" )

if [ -n "$DATA" ]; then
	if [ ! -d "$DATA" ]; then
		echo "Warning: data directory '$DATA' not found; viewer may still run if cameras.json exists."
	else
		CMD+=( -s "$(readlink -f "$DATA")" )
	fi
fi

if [ -n "$WIDTH" ] && [ -n "$HEIGHT" ]; then
	CMD+=( --rendering-size "$WIDTH" "$HEIGHT" )
fi

echo "Launching SIBR viewer:"
printf '  %q' "${CMD[@]}"
echo

exec "$(dirname "$0")/../gpu_env.sh" "${CMD[@]}"
