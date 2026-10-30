#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
GPUS="${GPUS:-1}"
if [ "$#" -ge 2 ] && [[ "$1" != --* ]]; then
  METADATA="$1"
  OUTPUT="$2"
  shift 2
  exec accelerate launch --num_processes "$GPUS" "$ROOT/training/train_remove.py"     --metadata_path "$METADATA" --output_dir "$OUTPUT" "$@"
fi
exec accelerate launch --num_processes "$GPUS" "$ROOT/training/train_remove.py" "$@"
