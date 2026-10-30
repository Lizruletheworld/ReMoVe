#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../../env.sh"
IMG=$(readlink -f "$1")
PROMPT="$2"
MESH=$(readlink -f "$3")
DEPTH=$(readlink -f "$4")
OUT=$(readlink -f "$5")
GPU=${6:-0}
CKPT=${7:-${WAN_LORA}}
FRAMES=${FRAMES:-auto}
FPS=${FPS:-15}
mkdir -p "$OUT"
NAME="$(basename "$IMG")"; NAME="${NAME%.*}"
activate_env remove
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="$GPU" python "$REMOVE_ROOT/scripts/wan/infer.py"   --input_image "$IMG" --mesh_video "$MESH" --depth_video "$DEPTH"   --prompt "$PROMPT" --output_dir "$OUT" --ckpt "$CKPT"   --num_frames "$FRAMES" --fps "$FPS" --device cuda:0
echo "[wan] -> $OUT/remove_wan.mp4"
