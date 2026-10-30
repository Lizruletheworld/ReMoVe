import sys, os, argparse, warnings
import numpy as np
import cv2
import torch

#  stub ?DA3  
from types import ModuleType as _MT
def _stub(name, **attrs):
    m = _MT(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules.setdefault(name, m)

_stub('moviepy.editor', VideoFileClip=None, concatenate_videoclips=None,
      ImageSequenceClip=None, ColorClip=None, CompositeVideoClip=None)
_stub('gsplat')
_stub('gsplat.rendering')
_stub('pycolmap')
_stub('pillow_heif')

#  DA3  
from depth_anything_3.api import DepthAnything3

def round16(x): return (x // 16) * 16

def load_video_bgr(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f": {path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    return frames

def parse_args():
    parser = argparse.ArgumentParser(description="Depth fusion with mesh conditioning")
    parser.add_argument("--raw_bg_video", required=True, help="original background video")
    parser.add_argument("--mesh_video_in", required=True, help="rendered mesh video")
    parser.add_argument("--bg_npz_in", required=True, help="static background-depth NPZ")
    parser.add_argument("--old_sam2_mask", required=True, help="original-person mask video")
    parser.add_argument("--mesh_mask", default=None, help="optional precise rendered-mesh mask video")
    parser.add_argument("--out_dir", default="./outputs/depth_test", help="output directory")
    parser.add_argument("--bg_color_bgr", type=int, nargs=3, default=[140, 140, 140])
    parser.add_argument("--mask_thresh", type=int, default=25)
    parser.add_argument("--mesh_mask_dilate", type=int, default=4)
    parser.add_argument("--mesh_mask_thresh", type=int, default=127)
    parser.add_argument("--min_visible", type=int, default=5)
    parser.add_argument("--inpaint_radius", type=int, default=7)
    parser.add_argument("--output_fps", type=int, default=24)
    parser.add_argument("--da3_batch", type=int, default=8)
    parser.add_argument("--da3_model", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--proxy", default=None, help="optional HTTP proxy for model download")
    return parser.parse_args()

def main():
    args = parse_args()

    # 
    out_comp_rgb = os.path.join(args.out_dir, "new_mesh_composite.mp4")
    out_depth_video = os.path.join(args.out_dir, "final_mesh_fused_depth_turbo.mp4")

    # 
    os.makedirs(args.out_dir, exist_ok=True)

    # 
    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy

    # 
    # Step 1: ?    # 
    print("[1] Loading all input source files...")
    raw_bg_frames = load_video_bgr(args.raw_bg_video)
    mesh_frames = load_video_bgr(args.mesh_video_in)

    with np.load(args.bg_npz_in) as data:
        old_raw_depth = data['depth'].astype(np.float32)

    # ask ?Step2 ?    old_mask_frames = load_video_bgr(args.old_sam2_mask)

    # ask ?Step3 ?    mesh_mask_frames = None
    if args.mesh_mask is not None:
        mesh_mask_frames = load_video_bgr(args.mesh_mask)
        print(f"    [ask] mask: {args.mesh_mask} ({len(mesh_mask_frames)} ?")
    else:
        print(f"    [ask] ?--mesh_mask (bg_color={args.bg_color_bgr}, thresh={args.mask_thresh})")

    T = len(mesh_frames)
    T_bg = len(raw_bg_frames)
    T_depth = len(old_raw_depth)
    T_mask = len(old_mask_frames)

    H, W = mesh_frames[0].shape[:2]
    OUT_W, OUT_H = round16(W), round16(H)
    print(f"    Target Duration (from PC video): {T} frames")

    # 
    # Step 2: ? remove_person_gpu.py ?    #   reconstruct_background
    #    coverage < min_visible ?inpaint ?    #   ?depth_bg.py ?inpaint
    #   ?"?
    # 
    print("[2] Reconstructing TEMPORAL-MEDIAN background depth (no per-frame ghosting)...")

    resized_old_depth = []
    resized_old_masks = []
    for i in range(T_depth):
        d_frame = cv2.resize(old_raw_depth[i], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        m_frame = old_mask_frames[min(i, T_mask - 1)]
        g_mask = cv2.cvtColor(cv2.resize(m_frame, (OUT_W, OUT_H)), cv2.COLOR_BGR2GRAY) > 127
        resized_old_depth.append(d_frame)
        resized_old_masks.append(g_mask)

    depth_stack = np.stack(resized_old_depth, axis=0).astype(np.float32)   # (T,H,W)
    mask_stack = np.stack(resized_old_masks, axis=0)                       # (T,H,W) bool, True=old person
    visible = ~mask_stack
    coverage = visible.sum(axis=0)                                         # 

    tmp = np.where(visible, depth_stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bg_depth_static = np.nanmedian(tmp, axis=0)
    bg_depth_static = np.nan_to_num(bg_depth_static, nan=0.0)

    unreliable = coverage < max(1, args.min_visible)
    print(f"    coverage: min={int(coverage.min())} mean={coverage.mean():.1f} "
          f"unreliable_px={int(unreliable.sum())} (of {coverage.size})")
    if unreliable.any():
        d_min, d_max = bg_depth_static.min(), bg_depth_static.max()
        bg_u8 = (((bg_depth_static - d_min) / (d_max - d_min + 1e-8)) * 255).astype(np.uint8)
        bg_u8 = cv2.inpaint(bg_u8, unreliable.astype(np.uint8) * 255,
                             inpaintRadius=args.inpaint_radius, flags=cv2.INPAINT_TELEA)
        bg_depth_static = (bg_u8.astype(np.float32) / 255.0) * (d_max - d_min) + d_min

    np.savez_compressed(
        os.path.join(args.out_dir, "bg_only_depth.npz"),
        depth=np.repeat(bg_depth_static[None], T_depth, axis=0).astype(np.float32),
    )

    # 
    # Step 3: RGB  + ask
    #   ??depth_bg.py ?    # 
    print("[3] Compositing new PC person onto background video...")
    kernel_mesh = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mesh_mask_dilate, args.mesh_mask_dilate))
                 if args.mesh_mask_dilate > 0 else None)
    comp_frames_rgb = []
    new_person_masks = np.zeros((T, OUT_H, OUT_W), dtype=bool)
    bg_color_np = np.array(args.bg_color_bgr, dtype=np.int16)
    T_meshmask = len(mesh_mask_frames) if mesh_mask_frames is not None else 0

    for i in range(T):
        idx_bg = min(i, T_bg - 1)
        bg_img = cv2.resize(raw_bg_frames[idx_bg], (OUT_W, OUT_H))
        mesh_img = cv2.resize(mesh_frames[i], (OUT_W, OUT_H))

        if mesh_mask_frames is not None:
            m_src = mesh_mask_frames[min(i, T_meshmask - 1)]
            m_gray = cv2.cvtColor(cv2.resize(m_src, (OUT_W, OUT_H)), cv2.COLOR_BGR2GRAY)
            m_mesh = (m_gray > args.mesh_mask_thresh).astype(np.uint8)
        else:
            diff = np.max(np.abs(mesh_img.astype(np.int16) - bg_color_np), axis=-1)
            m_mesh = (diff > args.mask_thresh).astype(np.uint8)

        if kernel_mesh is not None:
            m_mesh = cv2.dilate(m_mesh, kernel_mesh)
        new_person_masks[i] = m_mesh > 0

        m_mesh_3ch = m_mesh[:, :, None]
        comp = mesh_img.astype(np.float32) * m_mesh_3ch + bg_img.astype(np.float32) * (1 - m_mesh_3ch)
        comp_frames_rgb.append(comp.clip(0, 255).astype(np.uint8))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tw = cv2.VideoWriter(out_comp_rgb, fourcc, args.output_fps, (OUT_W, OUT_H))
    for f in comp_frames_rgb: tw.write(f)
    tw.release()

    # 
    # Step 4: ?DA3 ???    # 
    print(f"[4] Loading DA3 ({args.da3_model}) to estimate new person depth...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(args.da3_model).to(device).eval()

    print("    Running inference on composite frames...")
    new_comp_depth = []
    for start in range(0, T, args.da3_batch):
        batch_bgr = comp_frames_rgb[start:start + args.da3_batch]
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]
        with torch.no_grad():
            pred = model.inference(batch_rgb)
        for d in pred.depth:
            new_comp_depth.append(cv2.resize(d, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR))
    new_comp_depth = np.stack(new_comp_depth, axis=0).astype(np.float32)

    # 
    # Step 5:  ??+ ?    #   ?dynamic_bg_depths[idx_depth]?    #    bg_depth_static inpaint?    # 
    print("[5] Fusing depths: Temporal-median background + fresh DA3 person depth...")
    fused_metric_depth = np.zeros_like(new_comp_depth)
    for i in range(T):
        frame_depth = bg_depth_static.copy()
        m = new_person_masks[i]
        frame_depth[m] = new_comp_depth[i][m]
        fused_metric_depth[i] = frame_depth

    # 
    # Step 5.5:  ??    # 
    mesh_video_name = os.path.splitext(os.path.basename(args.mesh_video_in))[0]
    npz_out_name = f"{mesh_video_name}_fused_depth_raw.npz"

    print(f"[5.5] Saving raw fused depth to {npz_out_name}...")
    np.savez_compressed(
        os.path.join(args.out_dir, npz_out_name),
        depth=fused_metric_depth.astype(np.float32),
        person_mask=new_person_masks,
    )

    # 
    # Step 6:  Turbo ???    # 
    print("[6] Rendering final Turbo Colormap video...")
    writer = cv2.VideoWriter(out_depth_video + ".tmp.mp4", fourcc, args.output_fps, (OUT_W, OUT_H))
    g_min, g_max = fused_metric_depth.min(), fused_metric_depth.max()

    for d in fused_metric_depth:
        norm = ((d - g_min) / (g_max - g_min + 1e-8) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        writer.write(colored)
    writer.release()
    os.rename(out_depth_video + ".tmp.mp4", out_depth_video)

    print(f"\n[?")
    print(f"  ? ?{out_depth_video}")

if __name__ == "__main__":
    main()

