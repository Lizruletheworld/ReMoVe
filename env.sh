#!/bin/bash
# Source this file before running any step:   source env.sh
#
# Machine-specific overrides (conda location, env prefixes, HF cache, proxy ...)
# go into env.local.sh next to this file (git-ignored). See env.local.sh.example.

export REMOVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# where the conda environments live (created by setup/install_envs.sh)
export REMOVE_ENVS="${REMOVE_ENVS:-$REMOVE_ROOT/envs}"
# model weights that are not inside a third-party repo
export REMOVE_WEIGHTS="${REMOVE_WEIGHTS:-$REMOVE_ROOT/weights}"
# HuggingFace cache (DA3, LLM2Vec, Kimodo, Llama-3 land here)
export HF_HOME="${HF_HOME:-$REMOVE_WEIGHTS/hf_cache}"

[ -f "$REMOVE_ROOT/env.local.sh" ] && source "$REMOVE_ROOT/env.local.sh"

# External project roots are configured in env.local.sh.
# They intentionally live outside this repository.

# locate conda
if [ -z "${CONDA_ROOT:-}" ]; then
  if [ -d "$REMOVE_ROOT/miniconda3" ]; then
    CONDA_ROOT="$REMOVE_ROOT/miniconda3"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(dirname "$(dirname "$(command -v conda)")")"
  fi
fi
export CONDA_ROOT

# All stage scripts run in the single environment named remove.
activate_env() {
  local name=remove
  if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "conda not found (CONDA_ROOT=$CONDA_ROOT). Run setup/install_miniconda.sh or set CONDA_ROOT in env.local.sh" >&2
    return 1
  fi
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  if [ -d "$REMOVE_ENVS/$name/conda-meta" ]; then
    conda activate "$REMOVE_ENVS/$name"                 # prefix created by setup/install_envs.sh
  elif [ -x "$REMOVE_ENVS/$name/bin/python" ]; then
    source "$REMOVE_ENVS/$name/bin/activate"            # plain venv / virtualenv
  elif conda activate "$name" 2>/dev/null; then
    :                                                     # named conda env (conda create -n remove)
  else
    echo "environment '$name' not found (neither $REMOVE_ENVS/$name nor a conda env called '$name'); see README" >&2
    return 1
  fi
}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REMOVE_ROOT:$REMOVE_ROOT/scripts:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1    # never let ~/.local packages leak into (or mask gaps in) the conda envs
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch >= 2.6 defaults torch.load to weights_only; GVHMR/SMIRK pickles need the old behaviour
export FFMPEG="${FFMPEG:-ffmpeg}"
