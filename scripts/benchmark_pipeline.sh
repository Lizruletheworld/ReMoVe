#!/bin/bash
# Single-GPU end-to-end benchmark of the whole pipeline (steps 2-9) on the bundled example.
# Meant to be submitted as a 1x80GB job:   bash /path/to/ReMoVe/scripts/benchmark_pipeline.sh
#
# Writes everything under outputs/bench_<date>/ and prints a per-step timing table at the end
# (also saved to outputs/bench_<date>/timing.txt). Fully offline once the weights are in place.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/env.sh"

STAMP=$(date +%Y%m%d_%H%M%S)
WORK="$ROOT/outputs/bench_$STAMP"
mkdir -p "$WORK"
LOG="$WORK/bench.log"

# no network inside a job: everything is cached (HF Hub, GVHMR, SAM3, SMIRK, and Wan2.2 files)
export YOLO_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ffmpeg: job images often lack it; fall back to the static binary bundled with imageio-ffmpeg (the configured environment)
if ! command -v ffmpeg >/dev/null 2>&1; then
  FF=$(python3 -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
  mkdir -p "$WORK/bin" && ln -sf "$FF" "$WORK/bin/ffmpeg" && export PATH="$WORK/bin:$PATH"
fi
export FFMPEG=ffmpeg

{
  echo "### bench start $(date '+%F %T')  host=$(hostname)  work=$WORK"
  echo "### GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n 1)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
  echo "### ffmpeg: $(command -v ffmpeg)"
} | tee "$LOG"

T0=$(date +%s)
"$ROOT/run_pipeline.sh" \
  --video  "$ROOT/examples/lunges/coarse.mp4" \
  --prompt "$(cat "$ROOT/examples/lunges/prompt.txt")" \
  --head   "$ROOT/examples/lunges/head.jpg" \
  --work   "$WORK" --gpu 0 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%T)" "$line"; done | tee -a "$LOG"
RC=${PIPESTATUS[0]}
T1=$(date +%s)

python3 - "$LOG" "$T0" "$T1" "$RC" <<'PY' | tee "$WORK/timing.txt"
import re, sys, datetime as dt
log, t0, t1, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
order = ["gvhmr", "segmentation", "depth", "reference", "constraints", "kimodo", "render", "depth_fusion", "wan"]
marks = {}
for line in open(log, errors="ignore"):
    m = re.match(r"^(\d\d:\d\d:\d\d) \[(step\d+[ab]?)\] -> ", line)
    if m:
        h, mi, s = map(int, m.group(1).split(":"))
        marks.setdefault(m.group(2), h * 3600 + mi * 60 + s)
start = dt.datetime.fromtimestamp(t0); prev = start.hour * 3600 + start.minute * 60 + start.second
print(f"{'step':8s} {'finished':>9s} {'duration':>9s}")
for k in order:
    if k in marks:
        t = marks[k]; d = (t - prev) % 86400
        print(f"{k:8s} {t//3600:02d}:{t%3600//60:02d}:{t%60:02d} {d//60:6d}m{d%60:02d}s")
        prev = t
    else:
        print(f"{k:8s} {'-':>9s} {'-':>9s}")
tot = t1 - t0
print(f"{'TOTAL':8s} {'':>9s} {tot//60:6d}m{tot%60:02d}s   exit={rc}")
PY
echo "### bench end $(date '+%F %T'), log: $LOG"
exit "$RC"
