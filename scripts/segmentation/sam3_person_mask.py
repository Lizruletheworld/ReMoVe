#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
import glob

import contextlib
import types

import cv2
import numpy as np
import torch


#   torch  roi_align,  torchvision C++ op(metax torchvision _C ABI ) 
def _roi_align_impl(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    import torch.nn.functional as F
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    N, C, H, W = input.shape
    if isinstance(boxes, (list, tuple)):
        boxes_list = list(boxes)
    else:
        if boxes.numel() == 0:
            boxes_list = [input.new_zeros((0, 4)) for _ in range(N)]
        else:
            boxes_list = [boxes[boxes[:, 0] == b][:, 1:] for b in range(N)]
    offset = 0.5 if aligned else 0.0
    iy = (torch.arange(oh, device=input.device, dtype=input.dtype) + 0.5) / oh
    ix = (torch.arange(ow, device=input.device, dtype=input.dtype) + 0.5) / ow
    outs = []
    for b, bx in enumerate(boxes_list):
        n = bx.shape[0]
        if n == 0:
            outs.append(input.new_zeros((0, C, oh, ow)))
            continue
        bx = bx.to(input.dtype) * spatial_scale
        x1 = bx[:, 0] - offset; y1 = bx[:, 1] - offset
        x2 = bx[:, 2] - offset; y2 = bx[:, 3] - offset
        xs = x1[:, None] + (x2 - x1)[:, None] * ix[None, :]   # (n, ow) 
        ys = y1[:, None] + (y2 - y1)[:, None] * iy[None, :]   # (n, oh)
        gx = (2.0 * xs + 1.0) / W - 1.0                       # grid_sample align_corners=False
        gy = (2.0 * ys + 1.0) / H - 1.0
        grid = torch.stack([
            gx[:, None, :].expand(n, oh, ow),
            gy[:, :, None].expand(n, oh, ow),
        ], dim=-1)                                            # (n, oh, ow, 2)
        feat = input[b:b + 1].expand(n, C, H, W)
        sampled = F.grid_sample(feat, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        outs.append(sampled)
    return torch.cat(outs, dim=0)


#   sam3  triton kernel 
def _nms_cpu_impl(ious, scores, iou_threshold=0.5):
    ious_np = ious.float().detach().cpu().numpy()
    scores_np = scores.float().detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order.item(0)
        kept.append(i)
        inds = np.where(ious_np[i, order[1:]] <= iou_threshold)[0]
        order = order[inds + 1]
    return torch.tensor(kept, dtype=torch.int64, device=scores.device)


def _cc_cpu_impl(input_tensor):
    dev = input_tensor.device
    if input_tensor.shape[0] == 0:
        z = torch.zeros_like(input_tensor)
        return z, z
    from sam3.perflib.connected_components import connected_components_cpu
    l, s = connected_components_cpu(input_tensor.detach().cpu())
    return l.to(dev), s.to(dev)


#   shim: sam3  torch>=2.2  API,  torch 2.1.2+metax  
def _install_torch21_shims():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except Exception:
        import enum
        class SDPBackend(enum.IntEnum):
            ERROR = -1; MATH = 0; FLASH_ATTENTION = 1; EFFICIENT_ATTENTION = 2; CUDNN_ATTENTION = 3
        @contextlib.contextmanager
        def sdpa_kernel(backends=None, **kwargs): yield
        m = types.ModuleType("torch.nn.attention")
        m.sdpa_kernel = sdpa_kernel; m.SDPBackend = SDPBackend
        sys.modules["torch.nn.attention"] = m

    if not hasattr(torch, "compiler"):
        torch.compiler = types.ModuleType("torch.compiler")
    if not hasattr(torch.compiler, "is_dynamo_compiling"):
        try:
            import torch._dynamo as _d
            torch.compiler.is_dynamo_compiling = lambda: _d.is_compiling()
        except Exception:
            torch.compiler.is_dynamo_compiling = lambda: False

    from torch.utils import _pytree as _pt
    if not hasattr(_pt, "register_pytree_node") and hasattr(_pt, "_register_pytree_node"):
        def _reg_pytree(cls, flatten_fn, unflatten_fn, *a, **k):
            return _pt._register_pytree_node(cls, flatten_fn, unflatten_fn)
        _pt.register_pytree_node = _reg_pytree

    import torch.library as _tl
    if not hasattr(_tl, "custom_op"):
        def _custom_op(name, fn=None, *, mutates_args=(), device_types=None, schema=None):
            def deco(f):
                f.register_fake = lambda fake_fn: fake_fn
                f.register_kernel = lambda *a, **k: (lambda g: g)
                return f
            return deco if fn is None else deco(fn)
        _tl.custom_op = _custom_op
    if not hasattr(_tl, "register_fake"):
        def _register_fake(name, fn=None, *a, **k):
            return (lambda g: g) if fn is None else fn
        _tl.register_fake = _register_fake

    def _edt_cv2(data):
        arr = data.detach().to("cpu").numpy()
        shp = arr.shape
        flat = arr.reshape(-1, shp[-2], shp[-1])
        out = np.empty_like(flat, dtype=np.float32)
        for i in range(flat.shape[0]):
            src = (flat[i] != 0).astype(np.uint8)
            out[i] = cv2.distanceTransform(src, cv2.DIST_L2, 0)
        return torch.from_numpy(out.reshape(shp)).to(device=data.device, dtype=torch.float32)

    _edt_mod = types.ModuleType("sam3.model.edt")
    _edt_mod.edt_triton = _edt_cv2
    sys.modules["sam3.model.edt"] = _edt_mod

    _nms_mod = types.ModuleType("sam3.perflib.triton.nms")
    _nms_mod.nms_triton = _nms_cpu_impl
    sys.modules["sam3.perflib.triton.nms"] = _nms_mod
    _cc_mod = types.ModuleType("sam3.perflib.triton.connected_components")
    _cc_mod.connected_components_triton = _cc_cpu_impl
    sys.modules["sam3.perflib.triton.connected_components"] = _cc_mod

    import torchvision.ops as _tvops
    _tvops.roi_align = _roi_align_impl
    _roi_mod = sys.modules.get("torchvision.ops.roi_align")
    if _roi_mod is not None:
        _roi_mod.roi_align = _roi_align_impl


if os.environ.get("SAM3_COMPAT_SHIMS", "0") == "1":
    # Only for platforms without triton / torchvision CUDA ops (e.g. MetaX GPUs, torch<2.2).
    _install_torch21_shims()

from paths import SAM3_BPE, SAM3_CKPT

DEFAULT_BPE = SAM3_BPE
DEFAULT_CKPT = SAM3_CKPT
DEFAULT_INPUT_DIR = None
DEFAULT_OUTPUT_DIR = "outputs/sam3"


def parse_args():
    p = argparse.ArgumentParser(
        description="Remove people from videos under an input root directory (Multi-GPU batch)."
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Input root directory containing videos",
    )
    p.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output root directory. Relative subdirectories are preserved.",
    )
    p.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help=(
            "Optional specific video paths relative to --input_dir. "
            "If omitted, all mp4 files in --input_dir will be recursively scanned."
        ),
    )
    p.add_argument("--prompt", default="person")
    p.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--mask_dilate", type=int, default=9)
    p.add_argument("--min_visible", type=int, default=5)
    p.add_argument("--denoise_coverage", type=int, default=12)
    p.add_argument("--inpaint_radius", type=int, default=7)
    p.add_argument("--bpe_path", default=DEFAULT_BPE)
    p.add_argument("--checkpoint_path", default=DEFAULT_CKPT)
    p.add_argument("--save_debug", action="store_true")
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--import_only", action="store_true")
    
    # 
    p.add_argument("--gpu_id", type=int, default=0, help="Current GPU index inside the process context")
    p.add_argument("--world_size", type=int, default=1, help="Total number of GPUs used for splitting")
    return p.parse_args()


def resolve_all_videos(input_dir, video_specs=None):
    """ video_specs 
     None input_dir  .mp4/.MP4 
    """
    input_root = os.path.realpath(os.path.expanduser(input_dir))
    jobs = []
    seen = set()

    #  A input_dir
    if video_specs is None:
        print(f"[] Scanning all MP4 files recursively in: {input_root}", flush=True)
        #  mp4
        found_files = (
            glob.glob(os.path.join(input_root, "**", "*.mp4"), recursive=True) +
            glob.glob(os.path.join(input_root, "**", "*.MP4"), recursive=True)
        )
        # 
        found_files = sorted(list(set([os.path.realpath(f) for f in found_files])))
        for in_path in found_files:
            rel_path = os.path.relpath(in_path, input_root)
            jobs.append((in_path, rel_path))
        return jobs

    #  B
    for spec in video_specs:
        spec = os.path.expandvars(os.path.expanduser(spec.strip()))
        if not spec:
            continue
        if os.path.isabs(spec):
            absolute_spec = os.path.realpath(spec)
            try:
                inside_root = os.path.commonpath([input_root, absolute_spec]) == input_root
            except ValueError:
                inside_root = False
            in_path = absolute_spec if inside_root else os.path.realpath(os.path.join(input_root, spec.lstrip("/\\")))
        else:
            in_path = os.path.realpath(os.path.join(input_root, spec))

        try:
            inside_root = os.path.commonpath([input_root, in_path]) == input_root
        except ValueError:
            inside_root = False
        if not inside_root or not os.path.isfile(in_path) or not in_path.lower().endswith(".mp4") or in_path in seen:
            continue

        seen.add(in_path)
        rel_path = os.path.relpath(in_path, input_root)
        jobs.append((in_path, rel_path))

    return jobs


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames: {path}")
    return frames, fps


def extract_masks_from_outputs(outputs, H, W):
    if outputs is None:
        return np.zeros((H, W), dtype=bool)
    m = None
    if isinstance(outputs, dict):
        m = outputs.get("out_binary_masks", outputs.get("masks", None))
    if m is None:
        return np.zeros((H, W), dtype=bool)
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 4:
        m = m[:, 0]
    if m.ndim == 2:
        m = m[None]
    union = np.zeros((H, W), dtype=bool)
    for mm in m:
        mm = mm.astype(bool)
        if mm.shape != (H, W):
            mm = cv2.resize(mm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        union |= mm
    return union


def build_predictor(args):
    from sam3.model_builder import build_sam3_predictor
    print(f"[GPU {args.gpu_id}] Building SAM3 predictor (version={args.version}, fa3=False) ...", flush=True)
    predictor = build_sam3_predictor(
        checkpoint_path=args.checkpoint_path,
        bpe_path=args.bpe_path,
        version=args.version,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
    )
    return predictor


def masks_for_video(predictor, video_path, prompt, confidence, H, W, probe=False):
    resp = predictor.handle_request({"type": "start_session", "resource_path": video_path})
    sid = resp["session_id"]
    try:
        predictor.handle_request({
            "type": "add_prompt", "session_id": sid,
            "frame_index": 0, "text": prompt, "output_prob_thresh": confidence,
        })
        masks = {}
        for out in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": sid,
        }):
            fi = out["frame_index"]
            outputs = out.get("outputs", out)
            if probe and not masks:
                keys = list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)
                print(f"[probe] frame {fi} outputs keys: {keys}", flush=True)
            masks[fi] = extract_masks_from_outputs(outputs, H, W)
            if probe:
                break
        return masks
    finally:
        predictor.handle_request({"type": "close_session", "session_id": sid})


def reconstruct_background(frames_np, masks, min_visible, inpaint_radius, denoise_coverage=12):
    T, H, W = masks.shape
    visible = ~masks
    coverage = visible.sum(axis=0)
    frames_f = frames_np.astype(np.float32)
    tmp = np.where(visible[:, :, :, None], frames_f, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bg = np.nanmedian(tmp, axis=0)
    bg = np.nan_to_num(bg, nan=0.0)

    unreliable = coverage < max(1, min_visible)
    if unreliable.any():
        bg_u8 = bg.clip(0, 255).astype(np.uint8)
        bg = cv2.inpaint(bg_u8, unreliable.astype(np.uint8) * 255,
                         inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA).astype(np.float32)
    return bg.clip(0, 255), coverage


def open_writer(path, fps, size):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        raise RuntimeError(f"cannot open writer {path}")
    return w


def process_video(predictor, in_path, relative_path, out_path, args):
    relative_path = os.path.normpath(relative_path)
    relative_dir = os.path.dirname(relative_path)
    filename = os.path.basename(relative_path)
    stem = os.path.splitext(filename)[0]
    
    video_out_dir = os.path.dirname(out_path)

    frames, src_fps = read_frames(in_path)
    fps = args.fps or src_fps
    H, W = frames[0].shape[:2]
    T = len(frames)

    mask_dict = masks_for_video(predictor, in_path, args.prompt, args.confidence, H, W, probe=args.probe)
    if args.probe:
        return None

    kernel = None
    if args.mask_dilate > 0:
        k = args.mask_dilate if args.mask_dilate % 2 else args.mask_dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    masks = np.zeros((T, H, W), dtype=bool)
    tot = 0
    for i in range(T):
        m = mask_dict.get(i, np.zeros((H, W), dtype=bool))
        if kernel is not None and m.any():
            m = cv2.dilate(m.astype(np.uint8), kernel) > 0
        masks[i] = m
        tot += int(m.sum())
    print(f"  [GPU {args.gpu_id}] masks: {len(mask_dict)} frames, avg px={tot // max(1,T)}", flush=True)

    frames_np = np.stack(frames, axis=0)
    bg, coverage = reconstruct_background(frames_np, masks, args.min_visible, args.inpaint_radius, args.denoise_coverage)
    empty = int((coverage == 0).sum())
    if empty:
        print(f"  [GPU {args.gpu_id}] [note] {empty} px never background-visible (inpaint filled)", flush=True)

    os.makedirs(video_out_dir, exist_ok=True)
    tmp = out_path + ".tmp.mp4"
    writer = open_writer(tmp, fps, (W, H))
    dbg_mask = dbg_ov = None
    if args.save_debug:
        dbg_mask = open_writer(os.path.join(video_out_dir, stem + "_mask.mp4"), fps, (W, H))
        dbg_ov = open_writer(os.path.join(video_out_dir, stem + "_overlay.mp4"), fps, (W, H))
    for i in range(T):
        out_f = frames[i].astype(np.float32).copy()
        out_f[masks[i]] = bg[masks[i]]
        writer.write(out_f.clip(0, 255).astype(np.uint8))
        if args.save_debug:
            dbg_mask.write(cv2.cvtColor(masks[i].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
            ov = frames[i].copy()
            g = np.zeros_like(ov); g[:] = (0, 255, 0)
            ov[masks[i]] = (0.5 * ov[masks[i]] + 0.5 * g[masks[i]]).astype(np.uint8)
            dbg_ov.write(ov)
    writer.release()
    if args.save_debug:
        dbg_mask.release(); dbg_ov.release()
    os.replace(tmp, out_path)
    print(f"  [GPU {args.gpu_id}] -> {out_path}", flush=True)
    return out_path


def main():
    args = parse_args()
    if args.import_only:
        try:
            from sam3.model_builder import build_sam3_predictor
            print("IMPORT OK: build_sam3_predictor available", flush=True)
        except Exception:
            traceback.print_exc()
        return

    # 1.  mp4 
    all_jobs = resolve_all_videos(args.input_dir, args.videos)
    total_videos = len(all_jobs)
    if total_videos == 0:
        print(f"[ERROR] No valid mp4 videos found under: {args.input_dir}", flush=True)
        sys.exit(1)

    # 2. 
    my_jobs = [job for i, job in enumerate(all_jobs) if i % args.world_size == args.gpu_id]
    print(f"\n[GPU {args.gpu_id}/{args.world_size}] Assigned {len(my_jobs)}/{total_videos} videos.", flush=True)
    
    if not my_jobs:
        print(f"[GPU {args.gpu_id}] No tasks assigned. Exiting.", flush=True)
        return

    # 3.  CUDA device 
    if torch.cuda.is_available():
        #  Shell  CUDA_VISIBLE_DEVICES  cuda:0
        torch.cuda.set_device(0)

    # 4.  SAM3 
    predictor = build_predictor(args)
    ok, fail = [], []

    # 5. 
    for idx, (in_path, rel_path) in enumerate(my_jobs, 1):
        # 
        relative_path = os.path.normpath(rel_path)
        relative_dir = os.path.dirname(relative_path)
        filename = os.path.basename(relative_path)
        stem = os.path.splitext(filename)[0]
        video_out_dir = os.path.join(args.output_dir, relative_dir)
        out_path = os.path.join(video_out_dir, stem + ".mp4")

        #   
        if os.path.exists(out_path):
            print(f"  [GPU {args.gpu_id}] [{idx}/{len(my_jobs)}] Skipped (Already processed): {rel_path}", flush=True)
            ok.append(out_path)
            continue

        print(f"\n [GPU {args.gpu_id}] [{idx}/{len(my_jobs)}] Processing: {rel_path}", flush=True)
        try:
            r = process_video(
                predictor,
                in_path,
                rel_path,
                out_path,
                args,
            )
            if r:
                ok.append(r)
        except Exception:
            fail.append(in_path)
            print(f"[FAIL GPU {args.gpu_id}] {in_path}\n{traceback.format_exc()}", flush=True)

    print(f"\n[ GPU {args.gpu_id} DONE ] success={len(ok)} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()