#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

SOURCE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/download_model_weights.sh --source huggingface" >&2
      exit 2
      ;;
  esac
done

[ "$SOURCE" = "huggingface" ] || {
  echo "Only the official Hugging Face source is supported." >&2
  echo "Usage: bash scripts/download_model_weights.sh --source huggingface" >&2
  exit 2
}

activate_env remove

python - "$REMOVE_WEIGHTS" <<'PY'
import sys
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import hf_hub_download, snapshot_download

weights = Path(sys.argv[1])
weights.mkdir(parents=True, exist_ok=True)

wan_dir = weights / "wan_base"
snapshot_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    local_dir=str(wan_dir),
)
print(f"Wan2.2 base model: {wan_dir}")

remove_dir = weights / "wan_lora"
remove_file = hf_hub_download(
    repo_id="lizirulestheworld/ReMoVe",
    filename="Wan2.2-TI2V-5B-remove_step-31110.safetensors",
    local_dir=str(remove_dir),
)
print(f"ReMoVe weights: {remove_file}")

bpe_file = weights / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
if not bpe_file.exists():
    bpe_file.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/facebookresearch/sam3/0570b3a/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    with urlopen(url) as response, bpe_file.open("wb") as handle:
        handle.write(response.read())
print(f"SAM3 BPE resource: {bpe_file}")
PY
