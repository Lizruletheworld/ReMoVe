#!/bin/bash
# Kimodo motion generation: text prompt + kinematic constraints -> SMPL-X motion with NVIDIA Kimodo.
#
# usage: ../kimodo/generate_motion.sh "<prompt>" <constraints.json> <out.npz> [tier|duration]
#   tier "high" (default) -> --duration 2.7  (81 frames @ 30 fps = the 81-frame Wan video)
#   tier "medium"/"low"   -> --duration 5 and trim to the constrained span (trim_motion.py)
#   a number              -> used as the duration directly, no trimming
#
# Needs HF access to nvidia/Kimodo-SMPLX-RP-v1 and meta-llama/Meta-Llama-3-8B-Instruct
# (export HF_TOKEN=... or `hf auth login`). First run downloads ~17 GB.
# Prints the path of the motion npz to use in step 7 as the last line.
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

PROMPT=$1; CJSON=$(readlink -f "$2"); OUT=$3; TIER=${4:-high}
OUT=${OUT%.npz}
case "$TIER" in
  high)        DUR=2.7; TRIM=0 ;;
  medium|low)  DUR=5;   TRIM=1 ;;
  *)           DUR=$TIER; TRIM=0 ;;
esac
FINAL="$OUT.npz"; [ "$TRIM" = 1 ] && FINAL="${OUT}_trimmed.npz"
[ -f "$FINAL" ] && { echo "[kimodo] exists, skip: $FINAL"; echo "$FINAL"; exit 0; }

activate_env remove
export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"   # don't probe for a text-encoder service
mkdir -p "$(dirname "$OUT")"
python -m kimodo.scripts.generate \
  --model nvidia/Kimodo-SMPLX-RP-v1 \
  "$PROMPT" \
  --duration "$DUR" \
  --constraints "$CJSON" \
  --output "$(readlink -f "$(dirname "$OUT")")/$(basename "$OUT")"
if [ "$TRIM" = 1 ]; then
  python "$REMOVE_ROOT/scripts/kimodo/trim_motion.py" "$OUT.npz" --out "$FINAL" --threshold_ratio 1.0 --tail_pad 0
fi
echo "[kimodo] -> $FINAL"
echo "$FINAL"
