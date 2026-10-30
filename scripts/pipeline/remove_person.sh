#!/bin/bash
# Person segmentation: SAM3 video segmentation of "person" + temporal-median background reconstruction.
#
# usage: remove_person.sh <video.mp4> <out_dir>
# output: <out_dir>/<stem>.mp4 (person removed), <stem>_mask.mp4, <stem>_overlay.mp4
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

VIDEO=$(readlink -f "$1"); OUT_DIR=$2
stem=$(basename "${VIDEO%.*}")
[ -f "$OUT_DIR/${stem}_mask.mp4" ] && { echo "[segmentation] exists, skip: $OUT_DIR/${stem}_mask.mp4"; exit 0; }

activate_env remove
python "$REMOVE_ROOT/scripts/segmentation/sam3_person_mask.py" \
  --input_dir "$(dirname "$VIDEO")" --videos "$(basename "$VIDEO")" \
  --output_dir "$OUT_DIR" --save_debug "${@:3}"
echo "[segmentation] -> $OUT_DIR/$stem.mp4 , ${stem}_mask.mp4"
