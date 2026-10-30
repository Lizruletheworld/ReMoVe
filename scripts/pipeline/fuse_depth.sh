#!/bin/bash
# Depth fusion: fuse the background depth (step 3b, old person removed with the SAM3 mask of step 3a)
#         with DA3 depth of the new mesh person (step 7) -> final depth condition video.
#
# usage: fuse_depth.sh <raw_video.mp4> <mesh_video.mp4> <bg_depth_maps.npz> <sam3_mask.mp4> <out_dir>
# output: <out_dir>/final_mesh_fused_depth_turbo.mp4 (depth), <out_dir>/new_mesh_composite.mp4 (RGB composite)
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

RAW=$(readlink -f "$1"); MESH=$(readlink -f "$2"); NPZ=$(readlink -f "$3"); SAMMASK=$(readlink -f "$4"); OUT=$5
[ -f "$OUT/final_mesh_fused_depth_turbo.mp4" ] && { echo "[depth fusion] exists, skip: $OUT"; exit 0; }
MESHMASK="${MESH%.mp4}_mask.mp4"
[ -f "$MESHMASK" ] || { echo "[depth fusion] no ${MESHMASK}, falling back to colour keying"; MESHMASK=""; }

activate_env remove
python "$REMOVE_ROOT/scripts/depth/fuse_depth.py" \
  --raw_bg_video "$RAW" --mesh_video_in "$MESH" --bg_npz_in "$NPZ" --old_sam2_mask "$SAMMASK" \
  ${MESHMASK:+--mesh_mask "$MESHMASK"} --out_dir "$OUT"
echo "[depth fusion] -> $OUT/final_mesh_fused_depth_turbo.mp4"
