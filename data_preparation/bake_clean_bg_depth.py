#!/usr/bin/env python3
import argparse
import warnings
import numpy as np
import cv2


def load_video_gray_masks(path, out_w, out_h):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f": {path}")
    masks = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (out_w, out_h)), cv2.COLOR_BGR2GRAY)
        masks.append(g > 127)
    cap.release()
    return masks


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bg_npz_in", required=True, help=" (wan_init_video/depth/*_depth_maps.npz)")
    ap.add_argument("--old_sam2_mask", required=True, help="SAM2 mask")
    ap.add_argument("--out_w", type=int, required=True, help=" (front_back_cacheW)")
    ap.add_argument("--out_h", type=int, required=True, help=" (front_back_cacheH)")
    ap.add_argument("--min_visible", type=int, default=5,
                     help="")
    ap.add_argument("--inpaint_radius", type=int, default=7)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    with np.load(args.bg_npz_in) as data:
        old_raw_depth = data["depth"].astype(np.float32)  # (T,H,W)

    old_masks = load_video_gray_masks(args.old_sam2_mask, args.out_w, args.out_h)
    T_depth = old_raw_depth.shape[0]
    T_mask = len(old_masks)

    resized_depth = [cv2.resize(old_raw_depth[i], (args.out_w, args.out_h), interpolation=cv2.INTER_LINEAR)
                      for i in range(T_depth)]
    depth_stack = np.stack(resized_depth, axis=0).astype(np.float32)
    mask_stack = np.stack([old_masks[min(i, T_mask - 1)] for i in range(T_depth)], axis=0)
    visible = ~mask_stack
    coverage = visible.sum(axis=0)

    tmp = np.where(visible, depth_stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bg_depth_static = np.nanmedian(tmp, axis=0)
    bg_depth_static = np.nan_to_num(bg_depth_static, nan=0.0)

    unreliable = coverage < max(1, args.min_visible)
    print(f"coverage: min={int(coverage.min())} mean={coverage.mean():.1f} "
          f"unreliable_px={int(unreliable.sum())} (of {coverage.size})")
    if unreliable.any():
        d_min, d_max = bg_depth_static.min(), bg_depth_static.max()
        bg_u8 = (((bg_depth_static - d_min) / (d_max - d_min + 1e-8)) * 255).astype(np.uint8)
        bg_u8 = cv2.inpaint(bg_u8, unreliable.astype(np.uint8) * 255,
                             inpaintRadius=args.inpaint_radius, flags=cv2.INPAINT_TELEA)
        bg_depth_static = (bg_u8.astype(np.float32) / 255.0) * (d_max - d_min) + d_min

    np.savez_compressed(args.out_npz, depth=bg_depth_static.astype(np.float32))
    print(f"[DONE] -> {args.out_npz}")


if __name__ == "__main__":
    main()
