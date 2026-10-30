#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"
missing=0
for path in "$ROOT/tools/GVHMR" "$ROOT/tools/SMIRK"; do
  if [ ! -d "$path" ]; then
    echo "[missing] $path; run setup/clone_tools.sh" >&2
    missing=1
  else
    echo "[ok] $path"
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 1
fi
