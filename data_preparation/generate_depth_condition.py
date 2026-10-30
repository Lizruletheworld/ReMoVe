#!/usr/bin/env python3
import sys, os, argparse
from types import ModuleType as _MT

import cv2
import numpy as np
import torch


#  stub DA3  ( extract_depth_single.py) 
def _stub(name, **attrs):
    m = _MT(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)


_stub("moviepy.editor", VideoFileClip=None, concatenate_videoclips=None,
      ImageSequenceClip=None, ColorClip=None, CompositeVideoClip=None)
_stub("gsplat"); _stub("gsplat.rendering"); _stub("pycolmap"); _stub("pillow_heif")

from depth_anything_3.api import DepthAnything3  # noqa: E402


DATASET   = os.environ.get("REMOVE_DATASET_ROOT")
if not DATASET:
    raise RuntimeError("Set REMOVE_DATASET_ROOT to the local Motion-X++ root.")
GM_ROOT   = os.path.join(DATASET, "motion", "mesh_recovery", "global_motion")
MESH_ROOT = os.environ.get("REMOVE_MESH_ROOT", os.path.join(DATASET, "mesh"))
BG_COLOR_BGR = np.array([140, 140, 140], dtype=np.int16)  #  BG_COLOR 

VIDEO_DIR_MAP = {
    "haa500":  "haa500",
    "idea400": "idea400_light",
    "humman":  "humman",
    "kungfu":  "kungfu",
    "music":   "music",
    "perform": "perform",
}


def round16(x):
    return (x // 16) * 16


def mask_key(cat, stem):
    return f"{cat}__{stem}"


def enumerate_samples():
    """ Phase A ;  [(cat, stem, orig_vpath, mesh_vpath), ...]"""
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
            mpath = os.path.join(MESH_ROOT, cat, f"{stem}.mp4")
            if os.path.isfile(vpath):
                samples.append((cat, stem, vpath, mpath))
    return samples


def load_video_bgr(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f": {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f": {path}")
    return frames


def da3_depth(model, frames_bgr, out_w, out_h, batch):
    """ BGR  DA3,  (T,out_h,out_w) float32 metric """
    depths = []
    for start in range(0, len(frames_bgr), batch):
        chunk = frames_bgr[start:start + batch]
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in chunk]
        with torch.inference_mode():
            pred = model.inference(rgb)
        for d in pred.depth:
            d = np.asarray(d, dtype=np.float32)
            depths.append(cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(depths, axis=0).astype(np.float32)


def process_one(model, args, cat, stem, orig_vpath, mesh_vpath):
    mask_path = os.path.join(args.mask_root, mask_key(cat, stem) + ".mp4")
    if not os.path.isfile(mask_path):
        print(f"  [skip-nomask] {cat}/{stem}:  Phase A mask {mask_path}", flush=True)
        return False
    if not os.path.isfile(mesh_vpath):
        print(f"  [skip-nomesh] {cat}/{stem}:  mesh {mesh_vpath}", flush=True)
        return False

    raw_bg   = load_video_bgr(orig_vpath)   # ()
    pc       = load_video_bgr(mesh_vpath)   # mesh ()
    real_msk = load_video_bgr(mask_path)    # Phase A  mask

    T      = len(pc)
    T_bg   = len(raw_bg)
    T_mask = len(real_msk)
    H, W   = pc[0].shape[:2]
    OUT_W, OUT_H = round16(W), round16(H)

    #  Step 1: DA3  () 
    old_raw_depth = da3_depth(model, raw_bg, OUT_W, OUT_H, args.da3_batch)
    T_depth = len(old_raw_depth)

    def real_mask_at(i):
        m = real_msk[min(i, T_mask - 1)]
        g = cv2.cvtColor(cv2.resize(m, (OUT_W, OUT_H)), cv2.COLOR_BGR2GRAY)
        return g > 127

    #  Step 2: ,  inpaint  
    dyn_bg = []
    for i in range(T_depth):
        d = old_raw_depth[i].copy()
        g = real_mask_at(i)
        if g.any():
            dmin, dmax = float(d.min()), float(d.max())
            u8 = (((d - dmin) / (dmax - dmin + 1e-8)) * 255).astype(np.uint8)
            u8 = cv2.inpaint(u8, g.astype(np.uint8) * 255, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            d = (u8.astype(np.float32) / 255.0) * (dmax - dmin) + dmin
        dyn_bg.append(d)

    #  Step 3: mesh mask()   mask ->  
    kernel_pc = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mesh_mask_dilate, args.mesh_mask_dilate))
                 if args.mesh_mask_dilate > 0 else None)
    comp_frames = []
    occ_masks = np.zeros((T, OUT_H, OUT_W), dtype=bool)
    for i in range(T):
        bg_img = cv2.resize(raw_bg[min(i, T_bg - 1)], (OUT_W, OUT_H))
        mesh_img = cv2.resize(pc[i], (OUT_W, OUT_H))

        # mesh :  >  ( mesh_mask)
        diff = np.max(np.abs(mesh_img.astype(np.int16) - BG_COLOR_BGR), axis=-1)
        m_pc = (diff > args.mesh_thresh).astype(np.uint8)
        if kernel_pc is not None:
            m_pc = cv2.dilate(m_pc, kernel_pc)
        m_pc = m_pc > 0

        if args.occlusion:
            occ = m_pc & real_mask_at(i)      # :  mesh
        else:
            occ = m_pc                         #  =  depth_bg 
        occ_masks[i] = occ

        m3 = occ[:, :, None].astype(np.float32)
        comp = mesh_img.astype(np.float32) * m3 + bg_img.astype(np.float32) * (1 - m3)
        comp_frames.append(comp.clip(0, 255).astype(np.uint8))

    #  Step 4: DA3 () 
    new_depth = da3_depth(model, comp_frames, OUT_W, OUT_H, args.da3_batch)

    #  Step 5:  (, ) 
    fused = np.zeros((T, OUT_H, OUT_W), dtype=np.float32)
    for i in range(T):
        fd = dyn_bg[min(i, T_depth - 1)].copy()
        m = occ_masks[i]
        fd[m] = new_depth[i][m]
        fused[i] = fd

    #  Step 6: Turbo  mp4 
    out_path = os.path.join(args.out_root, f"{stem}.mp4")
    if os.path.isfile(out_path):
        # 3  stem: , 
        out_path = os.path.join(args.out_root, f"{stem}__{cat}.mp4")
    tmp = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (OUT_W, OUT_H))
    gmin, gmax = float(fused.min()), float(fused.max())
    denom = max(gmax - gmin, 1e-8)
    for d in fused:
        norm = np.clip((d - gmin) / denom * 255.0, 0, 255).astype(np.uint8)
        writer.write(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO))
    writer.release()
    os.replace(tmp, out_path)
    print(f"  {cat}/{stem}: T={T} {OUT_W}x{OUT_H} occ_avg={int(occ_masks.reshape(T,-1).sum(1).mean())}px -> {out_path}",
          flush=True)
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Phase B: DA3 depth fusion for depth_mesh")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--out_root", default=f"{DATASET}/depth_mesh",
                   help="Turbo  mp4  (,  <stem>.mp4)")
    p.add_argument("--mask_root", default=f"{DATASET}/depth_mesh_realmask",
                   help="Phase A  mask ")
    p.add_argument("--mesh_thresh", type=int, default=25, help="mesh [140]")
    p.add_argument("--mesh_mask_dilate", type=int, default=4, help="mesh mask ")
    p.add_argument("--occlusion", dest="occlusion", action="store_true", default=True,
                   help="[] mesh  meshmask  ()")
    p.add_argument("--no-occlusion", dest="occlusion", action="store_false",
                   help=",  mesh-only ( depth_bg)")
    p.add_argument("--output_fps", type=int, default=24)
    p.add_argument("--da3_batch", type=int, default=16)
    p.add_argument("--da3_model", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--proxy", default="", help="Optional HTTP proxy")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help=" N  (0=, )")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    if args.proxy:
        os.environ.setdefault("http_proxy", args.proxy)
        os.environ.setdefault("https_proxy", args.proxy)

    samples = enumerate_samples()
    mine = [s for i, s in enumerate(samples) if i % args.num_gpus == args.gpu_id]
    if args.limit > 0:
        mine = mine[:args.limit]
    print(f"[shard {args.gpu_id}/{args.num_gpus}] total={len(samples)} mine={len(mine)} occlusion={args.occlusion}",
          flush=True)
    if not mine:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model] Loading DA3 {args.da3_model} on {device} ...", flush=True)
    model = DepthAnything3.from_pretrained(args.da3_model).to(device).eval()

    ok = fail = skip = 0
    for n, (cat, stem, ovp, mvp) in enumerate(mine, 1):
        out_path = os.path.join(args.out_root, f"{stem}.mp4")
        if os.path.isfile(out_path) and not args.overwrite:
            skip += 1
            continue
        try:
            r = process_one(model, args, cat, stem, ovp, mvp)
            ok += int(bool(r))
        except Exception as e:
            fail += 1
            import traceback
            print(f"  [FAIL] {cat}/{stem}: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"[shard {args.gpu_id}] done ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
