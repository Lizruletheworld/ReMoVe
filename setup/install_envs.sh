#!/bin/bash
# Creates the single conda environment used by the pipeline (under $REMOVE_ENVS, default <repo>/envs).
#
# The release uses one environment for all stages.
#   remove    py3.10  torch 2.7.1+cu126, pytorch3d 0.7.8 (built from source), GVHMR, Depth-Anything-3, SAM3,
#             Kimodo + transformers 5, SMIRK deps, mediapipe              (steps 2-9)
#
# usage: setup/install_envs.sh
# Creates the single "remove" environment under envs/remove.
# Set CONDA_CREATE_EXTRA in env.local.sh only for additional official conda channels.
set -euo pipefail
export PIP_INDEX_URL=https://pypi.org/simple
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1   # packages in ~/.local must not mask gaps in the envs (jobs run with a different HOME)
mkdir -p "$REMOVE_ENVS"
bash "$REMOVE_ROOT/setup/clone_tools.sh"
bash "$REMOVE_ROOT/setup/check_tools.sh"

CONDA_CREATE_EXTRA=""

mkenv() {  # mkenv <name> <python-version>
  local env="$REMOVE_ENVS/$1"
  [ -x "$env/bin/python" ] || conda create -y -p "$env" "python=$2" $CONDA_CREATE_EXTRA
  conda activate "$env"
  pip install -U pip wheel "setuptools<81"
}

fix_chumpy() {  # chumpy (SMPL .pkl loader) does not build in isolation and uses removed numpy aliases
  pip install --no-build-isolation chumpy
  python -c "import chumpy" 2>/dev/null || sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf$/from numpy import nan, inf/' \
      "$REMOVE_ENVS/$1/lib/python3.10/site-packages/chumpy/__init__.py"
}

install_remove() {   # the single environment, same commands as README "Environment Setup"
  mkenv remove 3.10
  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1      # PyPI wheels are CUDA 12.6 builds
  pip install -r "$REMOVE_ROOT/requirements.txt"
  pip install -e "$GVHMR_ROOT" --no-deps
  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec
  pip install git+https://github.com/facebookresearch/sam3.git@0570b3a
  pip install git+https://github.com/nv-tlabs/kimodo.git@3b98f78
  pip install -e "$SMIRK_DIR" --no-deps
  fix_chumpy remove
  pip install "numpy==1.26.4" "setuptools<81"      # re-pin after the editable installs
  # pytorch3d has no prebuilt wheel for torch 2.7 -> build from source (10-25 min; needs nvcc 12.x)
  python -c "import pytorch3d" 2>/dev/null || \
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}" MAX_JOBS="${MAX_JOBS:-16}" \
    pip install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  python - <<'PY'
import torch, numpy, importlib
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__)
for m in ["pytorch3d", "hmr4d", "depth_anything_3", "sam3", "kimodo", "transformers", "peft", "diffusers", "accelerate", "safetensors", "sentencepiece", "mediapipe", "smplx", "chumpy", "xformers", "lightning", "timm", "ultralytics", "cv2", "PIL", "motion_correction"]:
    importlib.import_module(m); print("OK", m)
PY
  conda deactivate
}


[ "$#" -eq 0 ] || { echo "This release uses one environment; do not pass split or environment names." >&2; exit 2; }
install_remove
echo "installed under ${REMOVE_ENVS}/remove"
