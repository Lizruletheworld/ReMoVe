#!/bin/bash
# GVHMR fitting: GVHMR world-grounded SMPL-X fit of a video (static camera).
#
# usage: gvhmr.sh <video.mp4> <out_root>
# output: <out_root>/<video_stem>/hmr4d_results.pt  (+ incam / global preview videos)
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

VIDEO=$(readlink -f "$1"); OUT_ROOT=$(readlink -f "$2")
stem=$(basename "${VIDEO%.*}")
[ -f "$OUT_ROOT/$stem/hmr4d_results.pt" ] && { echo "[gvhmr] exists, skip: $OUT_ROOT/$stem/hmr4d_results.pt"; exit 0; }

activate_env remove
cd "$GVHMR_ROOT"        # demo.py resolves inputs/checkpoints relative to the repo root
python tools/demo/demo.py --video "$VIDEO" --output_root "$OUT_ROOT" -s
echo "[gvhmr] -> $OUT_ROOT/$stem/hmr4d_results.pt"
