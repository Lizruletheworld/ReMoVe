#!/bin/bash
# Mesh rendering: Kimodo motion -> skin-shaded SMPL body mesh with the user's real face (SMIRK/FLAME),
#         rendered in the target video's camera. Also writes <out>_mask.mp4.
#
# usage: render_mesh_video.sh <motion.npz> <target_hmr4d_results.pt> <ref_image> <ref_hmr4d_results.pt> <out.mp4>
#   motion.npz             Kimodo output (Kimodo) or a GVHMR fit converted with gvhmr_to_kimodo_gt.py
#   target_hmr4d_results   GVHMR fit of the Wan video (GVHMR): camera, resolution, first-frame alignment
#   ref_image              the user's front/head image (reference input)
#   ref_hmr4d_results      GVHMR fit of the 5-frame reference video (reference output)
#
# tunables (env): SHADE=0.45 HAIR=0.45 FEATHER=6 TEX=1024 OSCALE=2 COLOR="" (R,G,B or empty = sample skin from the face)
#                 SKIP_FACE=1  render without the SMIRK/FLAME face (no FLAME model needed; body colour defaults to light blue)
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

NPZ=$(readlink -f "$1"); TGT=$(readlink -f "$2"); IMG=$(readlink -f "$3"); REF=$(readlink -f "$4"); OUT=$5
[ -f "$OUT" ] && { echo "[render] exists, skip: $OUT"; exit 0; }
SHADE=${SHADE:-0.45}; HAIR=${HAIR:-0.45}; FEATHER=${FEATHER:-6}; TEX=${TEX:-1024}; OSCALE=${OSCALE:-2}; COLOR=${COLOR-}
mkdir -p "$(dirname "$OUT")"
stem=$(basename "${IMG%.*}")
LM="$(dirname "$OUT")/face_landmarks"

FACE_ARGS=()
if [ "${SKIP_FACE:-0}" = "1" ]; then
  [ -n "$COLOR" ] || COLOR=150,180,235
else
  # phase 1: MediaPipe 478-point face landmarks of the reference image (lightweight env)
  if [ ! -f "$LM/$stem.mplmk.npy" ]; then
    activate_env remove
    python "$REMOVE_ROOT/scripts/render/face_landmarks.py" "$IMG" --out_dir "$LM"
    conda deactivate
  fi
  FACE_ARGS=(--smirk_head_views "$IMG:$LM/$stem.mplmk.npy" --smirk_head_tex_size "$TEX" --face_texture_only
             --face_tex_hair_frac "$HAIR" --face_tex_feather "$FEATHER")
fi

# phase 2: render
activate_env remove
export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$REMOVE_ROOT/scripts:$PYTHONPATH"
cd "$REMOVE_ROOT/scripts/render"
python render_mesh_video.py \
  --kimodo "$NPZ" \
  --target_gvhmr "$TGT" --target_frame 0 \
  --front_img "$IMG" --front_gvhmr "$REF" \
  --output_scale "$OSCALE" \
  --skin_body --skin_shade "$SHADE" ${COLOR:+--skin_color $COLOR} \
  "${FACE_ARGS[@]}" \
  --out "$OUT"
echo "[render] -> $OUT (+ ${OUT%.mp4}_mask.mp4)"
