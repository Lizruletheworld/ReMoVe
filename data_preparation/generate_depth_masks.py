#!/usr/bin/env python3
import sys, os, argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PREP_ROOT = os.environ.get("REMOVE_PREP_ROOT", str(Path(__file__).resolve().parent))
DATASET = os.environ.get("REMOVE_DATASET_ROOT")
if not DATASET:
    raise RuntimeError("Set REMOVE_DATASET_ROOT to the local Motion-X++ root.")
GM_ROOT = os.path.join(DATASET, "motion", "mesh_recovery", "global_motion")

# cat ->  video  ( run_v3_mesh_bg_all.py )
VIDEO_DIR_MAP = {
    "haa500":  "haa500",
    "idea400": "idea400_light",
    "humman":  "humman",
    "kungfu":  "kungfu",
    "music":   "music",
    "perform": "perform",
}

#  remove_person : import  torch2.1/metax  shim
sys.path.insert(0, PREP_ROOT)
import sam3_person_mask as rp  # noqa: E402


def enumerate_samples():
    """ Phase B  (cat  + stem ), 
     [(cat, stem, orig_video_path), ...]"""
    samples = []
    for cat in sorted(os.listdir(GM_ROOT)):
        cat_dir = os.path.join(GM_ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue
        vid_cat = VIDEO_DIR_MAP.get(cat, cat)
        vid_dir = os.path.join(DATASET, "video", vid_cat)
        for fn in sorted(os.listdir(cat_dir)):
            if not fn.endswith(".json"):
                continue
            stem = fn[:-5]
            vpath = os.path.join(vid_dir, f"{stem}.mp4")
            if os.path.isfile(vpath):
                samples.append((cat, stem, vpath))
    return samples


def mask_key(cat, stem):
    # cat  stem  ( 3 ) 
    return f"{cat}__{stem}"


def write_mask_video(path, masks_uint8, fps, size):
    tmp = path + ".tmp.mp4"
    w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"cannot open writer {path}")
    for m in masks_uint8:
        w.write(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
    w.release()
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description="Phase A: SAM3 real-person masks for depth_mesh")
    p.add_argument("--gpu_id", type=int, default=0, help=" shard  (0..num_gpus-1)")
    p.add_argument("--num_gpus", type=int, default=1, help="")
    p.add_argument("--mask_root", default=f"{DATASET}/depth_mesh_realmask",
                   help=" mask  ( <cat>__<stem>.mp4)")
    p.add_argument("--prompt", default="person")
    p.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--mask_dilate", type=int, default=9,
                   help="mask  ( remove_person_gpu_test )")
    p.add_argument("--fps", type=float, default=24.0, help="mask ")
    p.add_argument("--overwrite", action="store_true", help="")
    p.add_argument("--limit", type=int, default=0, help=" N  (0=, )")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.mask_root, exist_ok=True)

    samples = enumerate_samples()
    mine = [s for i, s in enumerate(samples) if i % args.num_gpus == args.gpu_id]
    if args.limit > 0:
        mine = mine[:args.limit]
    print(f"[shard {args.gpu_id}/{args.num_gpus}] total={len(samples)} mine={len(mine)}", flush=True)
    if not mine:
        return

    #  build_predictor  args
    ns = SimpleNamespace(
        checkpoint_path=rp.DEFAULT_CKPT,
        bpe_path=rp.DEFAULT_BPE,
        version=args.version,
    )
    predictor = rp.build_predictor(ns)

    kernel = None
    if args.mask_dilate > 0:
        k = args.mask_dilate if args.mask_dilate % 2 else args.mask_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    ok = fail = skip = 0
    for n, (cat, stem, vpath) in enumerate(mine, 1):
        out_path = os.path.join(args.mask_root, mask_key(cat, stem) + ".mp4")
        if os.path.isfile(out_path) and not args.overwrite:
            skip += 1
            continue
        try:
            frames, src_fps = rp.read_frames(vpath)
            H, W = frames[0].shape[:2]
            T = len(frames)
            fps = args.fps or src_fps

            mask_dict = rp.masks_for_video(predictor, vpath, args.prompt,
                                           args.confidence, H, W)
            masks = []
            tot = 0
            for i in range(T):
                m = mask_dict.get(i, np.zeros((H, W), dtype=bool))
                if kernel is not None and m.any():
                    m = cv2.dilate(m.astype(np.uint8), kernel) > 0
                mu = (m.astype(np.uint8) * 255)
                masks.append(mu)
                tot += int(m.sum())
            write_mask_video(out_path, masks, fps, (W, H))
            ok += 1
            print(f"  [{n}/{len(mine)}] {cat}/{stem}: {T}f {W}x{H} avgpx={tot // max(1, T)} -> {out_path}",
                  flush=True)
        except Exception as e:
            fail += 1
            print(f"  [FAIL] {cat}/{stem}: {type(e).__name__}: {e}", flush=True)

    print(f"[shard {args.gpu_id}] done ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
