#!/usr/bin/env bash
set -euo pipefail

if [[ ${1-} == "" ]]; then
  echo "Usage: tools/open_usdview.sh <stage-path>" >&2
  exit 64
fi

USD_ROOT_DEFAULT="$HOME/Downloads/usd_root"
USD_ROOT="${USD_ROOT:-$USD_ROOT_DEFAULT}"
USDVIEW_SCRIPT="$USD_ROOT/scripts/usdview.sh"

if [[ ! -d "$USD_ROOT" ]]; then
  echo "USD root directory not found at '$USD_ROOT'. Set USD_ROOT to the extracted usd_root folder." >&2
  exit 65
fi

if [[ ! -x "$USDVIEW_SCRIPT" ]]; then
  echo "Could not locate usdview launcher at '$USDVIEW_SCRIPT'. Verify your USD installation." >&2
  exit 66
fi

STAGE_PATH="$(realpath "$1")"
if [[ ! -f "$STAGE_PATH" ]]; then
  echo "USD stage '$STAGE_PATH' does not exist." >&2
  exit 67
fi

"$USDVIEW_SCRIPT" "$STAGE_PATH"
