#!/bin/bash
# Constraint selection: physical-plausibility metrics of the GVHMR fit -> quality tier -> Kimodo constraint JSON.
#
#   high   : all frames are used as full-body constraints (Kimodo reproduces the video motion)
#   medium : only clean continuous segments are constrained
#   low    : only the first frames are constrained (Kimodo generates most of the motion from text)
#
# usage: select_constraints.sh <gvhmr_out_root> <work_dir> [--force-high]
# output: <work_dir>/csv/{<sample>_per_frame.csv, summary.csv}
#         <work_dir>/constraints/<sample>.json , <work_dir>/constraints/tiers.json
set -euo pipefail
source "$(dirname "$0")/../../env.sh"

GV=$(readlink -f "$1"); W=$2; FORCE=${3:-}
activate_env remove
if [ "$FORCE" = "--force-high" ]; then
  python "$REMOVE_ROOT/scripts/kimodo/force_high.py" --gvhmr_dir "$GV"/*/ --out_dir "$W/constraints"
else
  python "$REMOVE_ROOT/scripts/kimodo/eval_physical.py" --data_root "$GV" --out_dir "$W/csv"
  python "$REMOVE_ROOT/scripts/kimodo/quality_select.py" --gvhmr_root "$GV" --csv_root "$W/csv" --out_dir "$W/constraints"
fi
echo "[constraints] -> $W/constraints (tiers.json)"
