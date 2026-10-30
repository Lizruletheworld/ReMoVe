#!/bin/bash
# Installs a private Miniconda under <repo>/miniconda3 (skip if you already have conda:
# just set CONDA_ROOT in env.local.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh}"
[ -x "$ROOT/miniconda3/bin/conda" ] && { echo "already installed: $ROOT/miniconda3"; exit 0; }
TMP="${TMPDIR:-/tmp}/miniconda_installer.sh"
curl -L -o "$TMP" "$URL"
bash "$TMP" -b -p "$ROOT/miniconda3"
"$ROOT/miniconda3/bin/conda" --version
echo "installed to $ROOT/miniconda3 (env.sh picks it up automatically)"
