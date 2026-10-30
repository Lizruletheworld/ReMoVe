#!/usr/bin/env python3
import argparse, os, sys, json, pickle, time
import cv2
import numpy as np
import torch
from pathlib import Path

DATASET_ROOT = os.environ.get("REMOVE_DATASET_ROOT")
if not DATASET_ROOT:
    raise RuntimeError("Set REMOVE_DATASET_ROOT to the local Motion-X++ root.")
GVHMR_ROOT   = os.environ.get("GVHMR_ROOT")
if not GVHMR_ROOT:
    raise RuntimeError("Set GVHMR_ROOT to the external GVHMR checkout.")
SMPL_PATH    = os.path.join(GVHMR_ROOT, "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl")
OUT_ROOT     = os.environ.get("REMOVE_MESH_ROOT", os.path.join(DATASET_ROOT, "mesh"))
OUTPUT_FPS   = 24
SUBSAMPLE    = 2
HAIR_FRAC    = 0.35          # drop top fraction of head height (hair/crown) from the face region
SKIN_SHADE   = 0.45          # normal-based form shading on the skin body (0=flat, like v9)

VIDEO_DIR_MAP = {
    "haa500":  "haa500",
    "idea400": "idea400_light",
    "humman": "humman",
    "kungfu": "kungfu",
    "music": "music",
    "perform": "perform",
}


def round16(x):
    return (x // 16) * 16


def collect_pairs():
    gm_root = Path(DATASET_ROOT) / "motion/mesh_recovery/global_motion"
    pairs = []
    for cat_dir in sorted(gm_root.iterdir()):
        cat = cat_dir.name
        vid_cat = VIDEO_DIR_MAP.get(cat, cat)
        vid_dir = Path(DATASET_ROOT) / "video" / vid_cat
        out_dir = Path(OUT_ROOT) / cat
        for jpath in sorted(cat_dir.glob("*.json")):
            vpath = vid_dir / f"{jpath.stem}.mp4"
            opath = out_dir / f"{jpath.stem}.mp4"
            if vpath.exists():
                pairs.append((jpath, vpath, opath))
    return pairs


def setup_pytorch3d_stub():
    import types
    def _aa(aa):
        angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis  = aa / angle
        c, s  = torch.cos(angle)[..., None], torch.sin(angle)[..., None]
        K_ = torch.zeros(*axis.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
        K_[...,0,1]=-axis[...,2]; K_[...,0,2]=axis[...,1]
        K_[...,1,0]=axis[...,2];  K_[...,1,2]=-axis[...,0]
        K_[...,2,0]=-axis[...,1]; K_[...,2,1]=axis[...,0]
        I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(*K_.shape)
        return I + s*K_ + (1-c)*(K_@K_)
    def _r6d(d6):
        a1, a2 = d6[..., :3], d6[..., 3:6]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = torch.nn.functional.normalize(a2-(b1*a2).sum(-1,keepdim=True)*b1, dim=-1)
        return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)
    _t = types.ModuleType("pytorch3d.transforms")
    _t.axis_angle_to_matrix = _aa
    _t.rotation_6d_to_matrix = _r6d
    _p = types.ModuleType("pytorch3d")
    _p.transforms = _t
    sys.modules.setdefault("pytorch3d", _p)
    sys.modules["pytorch3d.transforms"] = _t


def rasterize_faceidx(verts_cam, faces_np, H, W, K):
    """Dense (SOLID) rasterisation: return a per-pixel SMPL face index (-1 = background),
    painter's order (far first) so the front-most face wins each pixel."""
    proj   = (K @ verts_cam.T).T
    z_v    = proj[:, 2]
    uv     = proj[:, :2] / (z_v[:, None] + 1e-9)
    face_z = z_v[faces_np].mean(axis=1)
    valid  = np.all(z_v[faces_np] > 0.1, axis=1)
    order  = np.argsort(face_z)[::-1]
    img    = np.zeros((H, W), dtype=np.int32)
    for fi in order:
        if not valid[fi]:
            continue
        pts = np.round(uv[faces_np[fi]]).astype(np.int32).reshape(-1, 1, 2)
        xs, ys = pts[:, 0, 0], pts[:, 0, 1]
        if xs.max() < 0 or xs.min() >= W or ys.max() < 0 or ys.min() >= H:
            continue
        cv2.fillConvexPoly(img, pts, int(fi + 1))
    return img - 1


def process_one(jpath, vpath, opath, smplx_model, smplx2smpl, faces_np, head_vert,
                device, skin_override=None, face_only=False):
    with open(jpath) as fp:
        gm = json.load(fp)
    anns = gm["annotations"]
    T    = len(anns)
    cp   = anns[0]["cam_params"]
    fx, fy, cx, cy = cp["intrins"]
    K_out = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    cam_R = np.array(cp["cam_R"], dtype=np.float32)
    cam_T = np.array(cp["cam_T"], dtype=np.float32)
    OUT_W, OUT_H = round16(int(cx*2)), round16(int(cy*2))

    cap = cv2.VideoCapture(str(vpath))
    frames = []
    for _ in range(T):
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(
            cv2.resize(bgr, (OUT_W, OUT_H)), cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0)
    cap.release()
    T = min(T, len(frames))
    if T == 0:
        return False

    def get_verts_cam(ann):
        sp = ann["smplx_params"]
        params = {
            "global_orient": torch.tensor(sp["root_orient"], dtype=torch.float32).unsqueeze(0).to(device),
            "body_pose":     torch.tensor(sp["pose_body"],   dtype=torch.float32).unsqueeze(0).to(device),
            "transl":        torch.tensor(sp["trans"],       dtype=torch.float32).unsqueeze(0).to(device),
            "betas":         torch.tensor(sp["betas"],       dtype=torch.float32).unsqueeze(0).to(device),
        }
        with torch.no_grad():
            vw = torch.matmul(smplx2smpl, smplx_model(**params).vertices[0]).cpu().numpy()
        return (cam_R @ vw.T + cam_T[:, None]).T

    verts_cam0 = get_verts_cam(anns[0])

    # ---- per-SMPL-face FACE-REGION mask: head joint (15) + camera-facing + below hairline ----
    head_face = head_vert[faces_np].any(axis=1)
    n0 = np.cross(verts_cam0[faces_np[:,1]]-verts_cam0[faces_np[:,0]],
                  verts_cam0[faces_np[:,2]]-verts_cam0[faces_np[:,0]])
    front_face = n0[:,2] < 0                                   # camera-facing at frame 0
    _hy = verts_cam0[head_vert][:,1]
    _ytop, _ybot = float(_hy.min()), float(_hy.max())          # cam y-down: min=crown, max=chin
    fcy = verts_cam0[faces_np].mean(axis=1)[:,1]
    below = fcy > (_ytop + HAIR_FRAC*(_ybot-_ytop))
    is_face_region = head_face & front_face & below            # the front-face patch

    opath.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = str(opath) + ".tmp.mp4"
    writer  = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"),
                              OUTPUT_FPS, (OUT_W, OUT_H))
    BG       = 0.55
    skin_rgb = np.asarray(skin_override, np.float32) if skin_override is not None else None

    for i in range(T):
        vt   = get_verts_cam(anns[i])
        fidx = rasterize_faceidx(vt, faces_np, OUT_H, OUT_W, K_out)   # SOLID dense raster
        # per-face facing (|normal.z|) for normal-based form shading on the body
        _fn  = np.cross(vt[faces_np[:,1]]-vt[faces_np[:,0]], vt[faces_np[:,2]]-vt[faces_np[:,0]])
        fnz  = np.abs(_fn[:,2]) / (np.linalg.norm(_fn, axis=1) + 1e-9)
        vy, vx = np.where(fidx >= 0)
        out  = np.full((OUT_H, OUT_W, 3), BG, dtype=np.float32)
        if len(vy):
            fpx = fidx[vy, vx]
            isf = is_face_region[fpx]                                 # face-region pixels
            # FACE -> real source-video texture (the mesh covers the person here)
            if isf.any():
                out[vy[isf], vx[isf]] = frames[i][vy[isf], vx[isf]]
            # body skin colour: sample once from the face pixels (per person), then reuse
            if skin_rgb is None and int(isf.sum()) > 50:
                _fc = frames[i][vy[isf], vx[isf]]
                _l = _fc.mean(1); _k = (_l>np.percentile(_l,20)) & (_l<np.percentile(_l,97))
                skin_rgb = (np.median(_fc[_k],axis=0) if int(_k.sum())>5 else _fc.mean(0)).astype(np.float32)
            if not face_only:
                _sk = skin_rgb if skin_rgb is not None else np.array([0.80,0.62,0.52], np.float32)
                # BODY -> skin colour with normal-based light/dark shading (camera-facing bright)
                _bm = ~isf
                _shade = ((1.0-SKIN_SHADE) + SKIN_SHADE*fnz[fpx[_bm]])[:, None]
                out[vy[_bm], vx[_bm]] = _sk[None, :] * _shade
            # face_only: leave the body as background -> only the real face remains
        frame = (out[:,:,::-1]*255).clip(0,255).astype(np.uint8)     # RGB->BGR
        writer.write(frame)

    writer.release()
    os.rename(tmp_out, str(opath))
    print(f"  [face_region faces={int(is_face_region.sum())}, skin_rgb="
          f"{None if skin_rgb is None else [round(float(x),3) for x in skin_rgb]}] -> {opath}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",   type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=6)
    parser.add_argument("--limit",    type=int, default=0, help="only process first N of this shard (0=all)")
    parser.add_argument("--skin_color", default=None, help='override body skin "R,G,B" 0-255 (RGB)')
    parser.add_argument("--face_only", action="store_true",
                        help="render ONLY the face (real texture); drop the body entirely "
                             "(-> output_v2_faceonly/ instead of output_v2_faceskin/)")
    parser.add_argument("--dry_run",  action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, GVHMR_ROOT)
    setup_pytorch3d_stub()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_pairs = collect_pairs()
    my_pairs  = [p for i, p in enumerate(all_pairs) if i % args.num_gpus == args.gpu_id]
    if args.face_only:   # redirect output dir so face-only runs don't clash with faceskin
        my_pairs = [(j, v, Path(str(o).replace("/output_v2_faceskin/", "/output_v2_faceonly/")))
                    for j, v, o in my_pairs]
    if args.limit > 0:
        my_pairs = my_pairs[:args.limit]
    print(f"[GPU {args.gpu_id}] total={len(all_pairs)}, my_share={len(my_pairs)}", flush=True)

    if args.dry_run:
        for j, v, o in my_pairs[:5]:
            print(f"  {j.name} -> {o}")
        return

    skin_override = None
    if args.skin_color:
        skin_override = np.array([float(x) for x in args.skin_color.split(',')], np.float32)[:3] / 255.0

    from hmr4d.utils.smplx_utils import make_smplx
    smplx_model = make_smplx("supermotion").eval().to(device)
    smplx2smpl  = torch.load(
        os.path.join(GVHMR_ROOT, "hmr4d/utils/body_model/smplx2smpl_sparse.pt"),
        map_location=device)
    with open(SMPL_PATH, "rb") as f:
        _smpl = pickle.load(f, encoding="latin1")
    faces_np  = np.array(_smpl["f"], dtype=np.int32)
    head_vert = (np.asarray(_smpl["weights"]).argmax(axis=1) == 15)   # SMPL joint 15 = head

    done = skipped = errors = 0
    t0 = time.time()
    for idx, (jpath, vpath, opath) in enumerate(my_pairs):
        if opath.exists():
            skipped += 1; continue
        try:
            process_one(jpath, vpath, opath, smplx_model, smplx2smpl, faces_np, head_vert,
                        device, skin_override=skin_override, face_only=args.face_only)
            done += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[GPU {args.gpu_id}] ERROR {jpath.name}: {e}", flush=True)
            errors += 1; continue
        print(f"[GPU {args.gpu_id}] {done+skipped}/{len(my_pairs)} done={done} skip={skipped} "
              f"err={errors}  {jpath.stem}", flush=True)

    print(f"[GPU {args.gpu_id}] DONE done={done} skipped={skipped} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
