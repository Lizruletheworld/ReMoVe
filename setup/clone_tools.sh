#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/tools"
if [ ! -d "$ROOT/tools/GVHMR/.git" ]; then
  git clone https://github.com/zju3dv/GVHMR.git "$ROOT/tools/GVHMR"
else
  echo "[skip] GVHMR already exists"
fi
git -C "$ROOT/tools/GVHMR" checkout ee960bb
if [ ! -d "$ROOT/tools/SMIRK/.git" ]; then
  git clone https://github.com/georgeretsi/smirk.git "$ROOT/tools/SMIRK"
else
  echo "[skip] SMIRK already exists"
fi
git -C "$ROOT/tools/SMIRK" checkout c7de404
