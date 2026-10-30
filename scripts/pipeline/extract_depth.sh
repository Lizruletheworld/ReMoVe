#!/bin/bash
# Depth extraction: metric depth of every frame with Depth-Anything-3 (DA3METRIC-LARGE).
#
# usage: extract_depth.sh <video.mp4 | dir_of_mp4> <out_dir> [batch=16]
# output: <out_dir>/<stem>_depth_maps.npz (float32 metres, key "depth"), <stem>_depth.mp4 (turbo preview)
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

IN=$(readlink -f "$1"); OUT_DIR=$2; BATCH=${3:-16}
if [ -f "$IN" ]; then
  stem=$(basename "${IN%.*}")
  [ -f "$OUT_DIR/${stem}_depth_maps.npz" ] && { echo "[depth] exists, skip: $OUT_DIR/${stem}_depth_maps.npz"; exit 0; }
fi

activate_env remove
python "$REMOVE_ROOT/scripts/depth/extract_depth.py" --video_dir "$IN" --out_dir "$OUT_DIR" --batch "$BATCH"
echo "[depth] -> $OUT_DIR"
