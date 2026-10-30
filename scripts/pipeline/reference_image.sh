#!/bin/bash
# Reference fitting: the user's high-resolution front/head reference image -> 5-frame still video -> GVHMR fit.
# (GVHMR's tracker needs >=3 consecutive frames to confirm a track, hence 5 identical frames.)
# The fit is what step 7 uses to align the SMIRK face and the body texture to the Kimodo mesh.
#
# usage: reference_image.sh <image.png> <out_root>
# output: <out_root>/<stem>_5frames.mp4 , <out_root>/gvhmr/<stem>_5frames/hmr4d_results.pt
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

IMG=$(readlink -f "$1"); OUT=$2
stem=$(basename "${IMG%.*}")
mkdir -p "$OUT"
REF_VIDEO="$OUT/${stem}_5frames.mp4"
if [ ! -f "$REF_VIDEO" ]; then
  "$FFMPEG" -y -loglevel error -loop 1 -i "$IMG" -frames:v 5 \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p -r 30 "$REF_VIDEO"
fi
bash "$REMOVE_ROOT/scripts/pipeline/gvhmr.sh" "$REF_VIDEO" "$OUT/gvhmr"
echo "[reference] -> $OUT/gvhmr/${stem}_5frames/hmr4d_results.pt"
