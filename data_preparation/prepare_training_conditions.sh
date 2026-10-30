#!/usr/bin/env bash
# Build ReMoVe training conditions from a local Motion-X++ checkout.
# Usage: bash prepare_training_conditions.sh <motionx_root> [num_gpus] [mesh_root] [mask_root] [depth_root]
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: $0 <motionx_root> [num_gpus] [mesh_root] [mask_root] [depth_root]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ROOT="$(readlink -f "$1")"
NUM_GPUS=1
[ "$#" -ge 2 ] && NUM_GPUS="$2"
MESH_ROOT="$DATASET_ROOT/mesh"
MASK_ROOT="$DATASET_ROOT/person_masks"
DEPTH_ROOT="$DATASET_ROOT/depth"
[ "$#" -ge 3 ] && MESH_ROOT="$(readlink -f "$3")"
[ "$#" -ge 4 ] && MASK_ROOT="$(readlink -f "$4")"
[ "$#" -ge 5 ] && DEPTH_ROOT="$(readlink -f "$5")"

MESH_PYTHON="$(printenv MESH_PYTHON 2>/dev/null || echo python3)"
SAM3_PYTHON="$(printenv SAM3_PYTHON 2>/dev/null || echo python3)"
DEPTH_PYTHON="$(printenv DEPTH_PYTHON 2>/dev/null || echo python3)"
GVHMR_ROOT="${GVHMR_ROOT:?Set GVHMR_ROOT to the external GVHMR checkout}"
export REMOVE_DATASET_ROOT="$DATASET_ROOT"
export REMOVE_MESH_ROOT="$MESH_ROOT"
export REMOVE_PREP_ROOT="$ROOT/data_preparation"
export GVHMR_ROOT
mkdir -p "$MESH_ROOT" "$MASK_ROOT" "$DEPTH_ROOT" "$ROOT/logs"

run_shards() {
  local label="$1"
  local python_bin="$2"
  local script="$3"
  shift 3
  local pids=""
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    echo "[$label] GPU $gpu/$NUM_GPUS"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$ROOT/data_preparation/$script" \
      --gpu_id "$gpu" --num_gpus "$NUM_GPUS" "$@" \
      >"$ROOT/logs/$label"_"$gpu.log" 2>&1 &
    pids="$pids $!"
  done
  local failed=0
  for pid in $pids; do
    wait "$pid" || failed=1
  done
  if [ "$failed" -ne 0 ]; then
    echo "[$label] failed; inspect $ROOT/logs/$label_*.log" >&2
    return 1
  fi
}
echo "[1/3] Rendering mesh conditions"
run_shards mesh "$MESH_PYTHON" generate_mesh_condition.py
echo "[2/3] Segmenting people with SAM3"
run_shards masks "$SAM3_PYTHON" generate_depth_masks.py --mask_root "$MASK_ROOT"
echo "[3/3] Fusing depth conditions"
run_shards depth "$DEPTH_PYTHON" generate_depth_condition.py --mask_root "$MASK_ROOT" --out_root "$DEPTH_ROOT"
echo "Done."
echo "  mesh : $MESH_ROOT"
echo "  mask : $MASK_ROOT"
echo "  depth: $DEPTH_ROOT"