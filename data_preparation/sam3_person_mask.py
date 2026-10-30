#!/usr/bin/env python3
import argparse
import os
import sys
import traceback

import contextlib
import types

import cv2
import numpy as np
import torch


#   torch  roi_align,  torchvision C++ op(metax torchvision _C ABI ) 
def _roi_align_impl(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    """ F.grid_sample  roi_align; bin ( sampling_ratio=1), aligned 
    boxes: List[Tensor(n,4)] ()  Tensor(K,5)([batch_idx,x1,y1,x2,y2]) (,C,oh,ow)"""
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


#   sam3  triton kernel(metax triton  libtinfo.so.5  kernel API ) 
def _nms_cpu_impl(ious, scores, iou_threshold=0.5):
    """generic NMS:  pairwise IoU  + scores,  index(,  sam3 generic_nms_cpu)"""
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
    """connected components:  sam3  CPU ,  device ;  batch()"""
    dev = input_tensor.device
    if input_tensor.shape[0] == 0:
        z = torch.zeros_like(input_tensor)
        return z, z
    from sam3.perflib.connected_components import connected_components_cpu
    l, s = connected_components_cpu(input_tensor.detach().cpu())
    return l.to(dev), s.to(dev)


#   shim: sam3  torch>=2.2  API,  torch 2.1.2+metax  
def _install_torch21_shims():
    # 1) torch.nn.attention.{sdpa_kernel, SDPBackend}  (torch>=2.2)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # noqa: F401
    except Exception:
        import enum

        class SDPBackend(enum.IntEnum):
            ERROR = -1
            MATH = 0
            FLASH_ATTENTION = 1
            EFFICIENT_ATTENTION = 2
            CUDNN_ATTENTION = 3

        @contextlib.contextmanager
        def sdpa_kernel(backends=None, **kwargs):
            # torch 2.1:  F.scaled_dot_product_attention , no-op 
            yield

        m = types.ModuleType("torch.nn.attention")
        m.sdpa_kernel = sdpa_kernel
        m.SDPBackend = SDPBackend
        sys.modules["torch.nn.attention"] = m

    # 2) torch.compiler.is_dynamo_compiling()  (torch>=2.3)
    if not hasattr(torch, "compiler"):
        torch.compiler = types.ModuleType("torch.compiler")
    if not hasattr(torch.compiler, "is_dynamo_compiling"):
        try:
            import torch._dynamo as _d
            torch.compiler.is_dynamo_compiling = lambda: _d.is_compiling()
        except Exception:
            torch.compiler.is_dynamo_compiling = lambda: False

    # 3) torch.utils._pytree.register_pytree_node  (torch>=2.2,  _register_pytree_node)
    from torch.utils import _pytree as _pt
    if not hasattr(_pt, "register_pytree_node") and hasattr(_pt, "_register_pytree_node"):
        def _reg_pytree(cls, flatten_fn, unflatten_fn, *a, **k):
            return _pt._register_pytree_node(cls, flatten_fn, unflatten_fn)
        _pt.register_pytree_node = _reg_pytree

    # 4) torch.library.custom_op / register_fake  (torch>=2.4)
    #     import ; use_fa3=False , no-op 
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

    # 5)  triton:  cv2.distanceTransform  sam3.model.edt.edt_triton
    #    edt.py  batched cv2.distanceTransform(x, DIST_L2, 0);
    #    metax triton  libtinfo.so.5  kernel API  -> 
    def _edt_cv2(data):
        # data: (B,H,W) binary tensor; data==0 (0), 0L2
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

    # 5b)  triton nms / connected_components (perflib/triton/__init__.py , )
    _nms_mod = types.ModuleType("sam3.perflib.triton.nms")
    _nms_mod.nms_triton = _nms_cpu_impl
    sys.modules["sam3.perflib.triton.nms"] = _nms_mod
    _cc_mod = types.ModuleType("sam3.perflib.triton.connected_components")
    _cc_mod.connected_components_triton = _cc_cpu_impl
    sys.modules["sam3.perflib.triton.connected_components"] = _cc_mod

    # 6)  torch  torchvision.ops.roi_align (metax torchvision _C ABI  torch)
    import torchvision.ops as _tvops
    _tvops.roi_align = _roi_align_impl
    _roi_mod = sys.modules.get("torchvision.ops.roi_align")
    if _roi_mod is not None:
        _roi_mod.roi_align = _roi_align_impl  #  RoIAlign 


_install_torch21_shims()

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BPE = os.environ.get(
    "SAM3_BPE",
    os.environ.get(
        "SAM3_BPE_PATH",
        os.path.join(os.environ.get("REMOVE_WEIGHTS", str(_ROOT / "weights")), "sam3", "bpe_simple_vocab_16e6.txt.gz"),
    ),
)
DEFAULT_CKPT = os.environ.get("SAM3_CHECKPOINT", str(_ROOT / "weights" / "sam3" / "sam3.pt"))
DEFAULT_INPUT_DIR = os.environ.get("REMOVE_INPUT_DIR", "")
DEFAULT_OUTPUT_DIR = os.environ.get("REMOVE_OUTPUT_DIR", "")


def parse_args():
    p = argparse.ArgumentParser(
        description="Remove people from explicitly specified videos under an input root directory."
    )
    p.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help="Input root directory, e.g. /home",
    )
    p.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output root directory. Relative subdirectories are preserved.",
    )
    p.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help=(
            "Video paths relative to --input_dir. Both '1/1.mp4' and '/1/1.mp4' "
            "are treated as paths inside --input_dir. Multiple paths are supported."
        ),
    )
    p.add_argument("--prompt", default="person")
    p.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--mask_dilate", type=int, default=9)
    p.add_argument("--min_visible", type=int, default=5)
    p.add_argument("--denoise_coverage", type=int, default=12,
                   help="coverage (0=)")
    p.add_argument("--inpaint_radius", type=int, default=7)
    p.add_argument("--bpe_path", default=DEFAULT_BPE)
    p.add_argument("--checkpoint_path", default=DEFAULT_CKPT)
    p.add_argument("--save_debug", action="store_true")
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--probe", action="store_true", help=" outputs ")
    p.add_argument("--import_only", action="store_true", help=" import sam3(shim),  traceback")
    return p.parse_args()


def resolve_videos(input_dir, video_specs):
    """Resolve explicitly listed videos under input_dir.

    Rules:
      - "1/1.mp4"  -> <input_dir>/1/1.mp4
      - "/1/1.mp4" -> <input_dir>/1/1.mp4
      - an absolute path already inside input_dir is accepted as-is

    Returns a list of (absolute_input_path, relative_path_under_input_dir).
    Missing files are warned and skipped. Duplicate paths are processed once.
    """
    input_root = os.path.realpath(os.path.expanduser(input_dir))
    jobs = []
    seen = set()

    for spec in video_specs:
        spec = os.path.expandvars(os.path.expanduser(spec.strip()))
        if not spec:
            continue

        # If an absolute path already points inside input_root, keep it.
        # Otherwise, a leading slash means "relative to input_root", as requested.
        if os.path.isabs(spec):
            absolute_spec = os.path.realpath(spec)
            try:
                inside_root = os.path.commonpath([input_root, absolute_spec]) == input_root
            except ValueError:
                inside_root = False

            if inside_root:
                in_path = absolute_spec
            else:
                in_path = os.path.realpath(
                    os.path.join(input_root, spec.lstrip("/\\"))
                )
        else:
            in_path = os.path.realpath(os.path.join(input_root, spec))

        # Prevent ../ from escaping the selected input root.
        try:
            inside_root = os.path.commonpath([input_root, in_path]) == input_root
        except ValueError:
            inside_root = False
        if not inside_root:
            print(f"[warn] path escapes input_dir, skipped: {spec}", flush=True)
            continue

        if not os.path.isfile(in_path):
            print(f"[warn] video not found, skipped: {in_path}", flush=True)
            continue

        if not in_path.lower().endswith(".mp4"):
            print(f"[warn] not an mp4 file, skipped: {in_path}", flush=True)
            continue

        if in_path in seen:
            print(f"[warn] duplicate video, skipped: {in_path}", flush=True)
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
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames: {path}")
    return frames, fps


def extract_masks_from_outputs(outputs, H, W):
    """,  out_binary_masks  union -> (H,W) bool"""
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
    print(f"Building SAM3 video predictor (version={args.version}, fa3=False) on GPU ...", flush=True)
    predictor = build_sam3_predictor(
        checkpoint_path=args.checkpoint_path,
        bpe_path=args.bpe_path,
        version=args.version,
        use_fa3=False,          # metax  flash-attn2,  fa3
        use_rope_real=True,
        compile=False,
        warm_up=False,
    )
    return predictor


def masks_for_video(predictor, video_path, prompt, confidence, H, W, probe=False):
    """ {frame_idx: mask(H,W)bool}"""
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
                if isinstance(outputs, dict):
                    for k, v in outputs.items():
                        shp = getattr(v, "shape", None)
                        print(f"          {k}: type={type(v).__name__} shape={shp}", flush=True)
            masks[fi] = extract_masks_from_outputs(outputs, H, W)
            if probe:
                break
        return masks
    finally:
        predictor.handle_request({"type": "close_session", "session_id": sid})


def reconstruct_background(frames_np, masks, min_visible, inpaint_radius, denoise_coverage=12):
    """: ' person '
    (coverage ),  inpaint """
    T, H, W = masks.shape
    visible = ~masks
    coverage = visible.sum(axis=0)
    frames_f = frames_np.astype(np.float32)
    #  nan,  nanmedian
    tmp = np.where(visible[:, :, :, None], frames_f, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bg = np.nanmedian(tmp, axis=0)          # (H,W,3);  nan
    bg = np.nan_to_num(bg, nan=0.0)

    # coverage () inpaint 
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


def process_video(predictor, in_path, relative_path, out_dir, args):
    relative_path = os.path.normpath(relative_path)
    relative_dir = os.path.dirname(relative_path)
    filename = os.path.basename(relative_path)
    stem = os.path.splitext(filename)[0]

    video_out_dir = os.path.join(out_dir, relative_dir)
    out_path = os.path.join(video_out_dir, stem + ".mp4")
    frames, src_fps = read_frames(in_path)
    fps = args.fps or src_fps
    H, W = frames[0].shape[:2]
    T = len(frames)
    print(f"\n=== {stem}: {T}f {W}x{H} @ {fps:.0f}fps ===", flush=True)

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
    print(f"  masks: {len(mask_dict)} frames, avg px={tot // max(1,T)}", flush=True)

    frames_np = np.stack(frames, axis=0)
    bg, coverage = reconstruct_background(frames_np, masks, args.min_visible, args.inpaint_radius, args.denoise_coverage)
    empty = int((coverage == 0).sum())
    if empty:
        print(f"  [note] {empty} px never background-visible (inpaint filled)", flush=True)

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
    print(f"  -> {out_path}", flush=True)
    return out_path


def main():
    args = parse_args()
    if args.import_only:
        try:
            from sam3.model_builder import build_sam3_predictor  # noqa
            print("IMPORT OK: build_sam3_predictor available", flush=True)
        except Exception:
            traceback.print_exc()
        return
    jobs = resolve_videos(args.input_dir, args.videos)
    if not jobs:
        print("no valid videos", flush=True)
        sys.exit(1)

    print(f"input root : {os.path.realpath(args.input_dir)}", flush=True)
    print(f"output root: {os.path.realpath(args.output_dir)}", flush=True)
    print(f"will process {len(jobs)} videos", flush=True)
    for in_path, rel_path in jobs:
        print(f"  {in_path} -> {os.path.join(args.output_dir, rel_path)}", flush=True)

    predictor = build_predictor(args)
    ok, fail = [], []
    for in_path, rel_path in jobs:
        try:
            r = process_video(
                predictor,
                in_path,
                rel_path,
                args.output_dir,
                args,
            )
            if r:
                ok.append(r)
        except Exception:
            fail.append(in_path)
            print(f"[FAIL] {in_path}\n{traceback.format_exc()}", flush=True)
    print(f"\nDone. success={len(ok)} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()
