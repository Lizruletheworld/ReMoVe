import os, sys, argparse
import numpy as np, cv2, torch
from scipy.spatial import cKDTree
from paths import GVHMR_ROOT, HY_ROOT, SMPLX_DIR, SMIRK_DIR, SMIRK_CKPT, DECA_DIR, HEAD_TEMPLATE_OBJ
KIMODO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIMODO_DIR)

# mediapipe(478) -> iBUG static idx (=iBUG_0idx-17), same as smirk_face.CORR
CORR = [(1, 13), (2, 16), (33, 19), (133, 22), (362, 25), (263, 28),
        (61, 31), (291, 37), (0, 34), (17, 40), (55, 4), (46, 0), (285, 5), (276, 9)]


def mock_p3d():
    import types
    def _aa(a):
        ang = a.norm(dim=-1, keepdim=True).clamp(min=1e-8); ax = a / ang
        c, s = torch.cos(ang)[..., None], torch.sin(ang)[..., None]
        K = torch.zeros(*ax.shape[:-1], 3, 3, device=a.device, dtype=a.dtype)
        K[..., 0, 1] = -ax[..., 2]; K[..., 0, 2] = ax[..., 1]; K[..., 1, 0] = ax[..., 2]
        K[..., 1, 2] = -ax[..., 0]; K[..., 2, 0] = -ax[..., 1]; K[..., 2, 1] = ax[..., 0]
        I = torch.eye(3, device=a.device, dtype=a.dtype).expand(*K.shape); return I + s * K + (1 - c) * (K @ K)
    p3 = types.ModuleType('pytorch3d.transforms'); p3.axis_angle_to_matrix = _aa; p3.rotation_6d_to_matrix = lambda d: d
    m = types.ModuleType('pytorch3d'); m.transforms = p3
    sys.modules.setdefault('pytorch3d', m); sys.modules['pytorch3d.transforms'] = p3


def _head_frame(L):
    """Robust head frame from iBUG-static-51 landmark GROUPS (averaged -> stable on
    low-poly SMPL). static idx = iBUG_0idx-17: R-eye 19..24, L-eye 25..30, mouth 31..50."""
    Reye = L[19:25].mean(0); Leye = L[25:31].mean(0); eye = (Reye + Leye) / 2
    mouth = L[31:51].mean(0)
    x = Leye - Reye; sx = np.linalg.norm(x) + 1e-9; x = x / sx
    y = mouth - eye; y = y - x * (y @ x); y = y / (np.linalg.norm(y) + 1e-9)
    z = np.cross(x, y); z = z / (np.linalg.norm(z) + 1e-9)
    return np.stack([x, y, z], 1), eye, sx   # frame(cols), center, interocular scale


def sim_procrustes(src, dst):
    """Similarity FLAME->SMPL via anatomical head frames (robust to low-poly)."""
    Fs, cs, ss = _head_frame(src); Fd, cd, sd = _head_frame(dst)
    s = sd / (ss + 1e-9); R = Fd @ Fs.T
    return s, R, cd - s * R @ cs


def sim_procrustes_svd(src, dst):
    """Similarity transform src->dst. Less opinionated than _head_frame alignment."""
    cs, cd = src.mean(0), dst.mean(0)
    S, D = src - cs, dst - cd
    U, _, Vt = np.linalg.svd(D.T @ S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy(); U[:, -1] *= -1; R = U @ Vt
    s = np.trace(R @ (S.T @ D)) / (np.sum(S * S) + 1e-9)
    return s, R, cd - s * R @ cs


def parse_flame_obj_uv(obj_path, faces):
    """Load FLAME vt/ft from an OBJ whose face order matches FLAME faces."""
    if not obj_path or not os.path.exists(obj_path):
        return None, None
    vt, ft, fv = [], [], []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("vt "):
                parts = line.strip().split()
                vt.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                vid, tid = [], []
                for tok in line.strip().split()[1:4]:
                    bits = tok.split("/")
                    vid.append(int(bits[0]) - 1)
                    tid.append(int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1)
                fv.append(vid); ft.append(tid)
    if not vt or len(ft) != len(faces):
        return None, None
    fv = np.asarray(fv, np.int64)
    if fv.shape != faces.shape or not np.array_equal(fv, faces.astype(np.int64)):
        print(f"[uv][WARN] {obj_path} face order does not match FLAME; UV atlas disabled")
        return None, None
    return np.asarray(vt, np.float32), np.asarray(ft, np.int32)


def bilinear_sample(img, xy):
    H, W = img.shape[:2]
    x = np.clip(xy[:, 0], 0, W - 1.001)
    y = np.clip(xy[:, 1], 0, H - 1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - wx) * (1 - wy) +
            img[y0, x1] * wx * (1 - wy) +
            img[y1, x0] * (1 - wx) * wy +
            img[y1, x1] * wx * wy)


def vnormals(V, F):
    nrm = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(nrm, F[:, k], fn)
    return nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)


def bake_view_dense(V0, F, lmk51, img_path, lmk_path):
    """Project V0 into one source view and keep enough data for UV-atlas baking."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    alpha = img[:, :, 3] if (img.ndim == 3 and img.shape[2] == 4) else None
    bgr = img[:, :, :3] if img.ndim == 3 else img
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    person = (alpha > 128) if alpha is not None else np.ones(rgb.shape[:2], bool)
    Hf, Wf = rgb.shape[:2]
    mp = np.load(lmk_path).astype(np.float32)
    obj = np.array([lmk51[si] for _, si in CORR], np.float64)
    img2d = np.array([mp[mi] for mi, _ in CORR], np.float64)
    f = 1.5 * max(Wf, Hf)
    Kc = np.array([[f, 0, Wf / 2], [0, f, Hf / 2], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img2d, Kc, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError(f"solvePnP failed for {img_path}")
    R, _ = cv2.Rodrigues(rvec)
    uv = cv2.projectPoints(V0.astype(np.float64), rvec, tvec, Kc, None)[0].reshape(-1, 2).astype(np.float32)
    vn_cam = (R @ vnormals(V0, F).T).T
    vert_vis = -vn_cam[:, 2].astype(np.float32)
    return {"image": rgb, "person": person, "uv": uv, "vert_vis": vert_vis}


def rasterize_uv_triangle(tex_shape, uv_tri):
    H, W = tex_shape
    p = uv_tri.copy()
    p[:, 0] *= (W - 1)
    p[:, 1] = (1.0 - p[:, 1]) * (H - 1)
    x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int)
    x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, W - 1); y1 = min(y1, H - 1)
    if x1 < x0 or y1 < y0:
        return None, None, None
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    px = xs.ravel() + 0.5; py = ys.ravel() + 0.5
    d = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) +
         (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
    if abs(d) < 1e-9:
        return None, None, None
    a = ((p[1, 1] - p[2, 1]) * (px - p[2, 0]) + (p[2, 0] - p[1, 0]) * (py - p[2, 1])) / d
    b = ((p[2, 1] - p[0, 1]) * (px - p[2, 0]) + (p[0, 0] - p[2, 0]) * (py - p[2, 1])) / d
    c = 1 - a - b
    m = (a >= 0) & (b >= 0) & (c >= 0)
    if not m.any():
        return None, None, None
    return xs.ravel()[m], ys.ravel()[m], np.stack([a[m], b[m], c[m]], 1).astype(np.float32)


def bake_uv_atlas(V0, F, vt, ft, face_ids, view_data, tex_size):
    tex = np.zeros((tex_size, tex_size, 3), np.float32)
    mask = np.zeros((tex_size, tex_size), np.uint8)
    for fi in face_ids:
        xs, ys, bc = rasterize_uv_triangle((tex_size, tex_size), vt[ft[fi]])
        if xs is None:
            continue
        tri = F[fi]
        src_xy = [np.einsum("mc,cd->md", bc, v["uv"][tri]) for v in view_data]
        scores = []
        good = []
        for vi, v in enumerate(view_data):
            H, W = v["person"].shape
            sx = np.clip(np.round(src_xy[vi][:, 0]).astype(np.int32), 0, W - 1)
            sy = np.clip(np.round(src_xy[vi][:, 1]).astype(np.int32), 0, H - 1)
            onp = v["person"][sy, sx]
            src_col = v["image"][sy, sx]
            not_bg = ~((src_col.min(1) > 0.85) | (src_col.max(1) < 0.06))
            sc = float(v["vert_vis"][tri].mean())
            scores.append(np.full(len(xs), sc, np.float32))
            good.append(onp & not_bg & (sc > 0.0))
        scores = np.stack(scores, 0)
        good = np.stack(good, 0)
        scores[~good] = -1e9
        best = np.argmax(scores, 0)
        valid = scores[best, np.arange(scores.shape[1])] > -1e8
        if not valid.any():
            continue
        col = np.zeros((len(xs), 3), np.float32)
        for vi, v in enumerate(view_data):
            m = valid & (best == vi)
            if m.any():
                col[m] = bilinear_sample(v["image"], src_xy[vi][m])
        tex[ys[valid], xs[valid]] = col[valid]
        mask[ys[valid], xs[valid]] = 255
    if (mask > 0).any():
        _, lbl = cv2.distanceTransformWithLabels((mask == 0).astype(np.uint8),
                                                 cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        yy, xx = np.where(mask > 0)
        fill = mask == 0
        idx = lbl[fill] - 1
        tex[fill] = tex[yy[idx], xx[idx]]
    print(f"[uv] baked {tex_size}x{tex_size} FLAME texture atlas, coverage={(mask>0).mean()*100:.1f}%")
    return tex


def sample_uv_atlas(tex, uv_xy):
    xy = uv_xy.copy()
    xy[:, 0] *= (tex.shape[1] - 1)
    xy[:, 1] = (1.0 - xy[:, 1]) * (tex.shape[0] - 1)
    return bilinear_sample(tex, xy)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_smpl_lmk_vids(target_gvhmr, target_frame, cache_verts0):
    mock_p3d()
    sys.path.insert(0, GVHMR_ROOT)
    if HY_ROOT:
        sys.path.insert(0, HY_ROOT)
    import smplx
    from smplx.lbs import vertices2landmarks
    from hmr4d.utils.smplx_utils import make_smplx
    dev = "cpu"
    smplx_model = make_smplx("supermotion").eval().to(dev)
    smplx2smpl = torch.load(os.path.join(GVHMR_ROOT, "hmr4d/utils/body_model/smplx2smpl_sparse.pt"),
                            map_location=dev)
    std = smplx.SMPLX(model_path=SMPLX_DIR,
                      use_face_contour=False, use_pca=False, batch_size=1)
    pred = torch.load(target_gvhmr, map_location="cpu", weights_only=False)
    par = {k: v[target_frame:target_frame + 1].float() for k, v in pred["smpl_params_incam"].items()}
    with torch.no_grad():
        vsx = smplx_model(**par).vertices
        vsmpl_canon = torch.matmul(smplx2smpl, vsx[0]).numpy()
        sxlmk = vertices2landmarks(vsx, std.faces_tensor,
                                   std.lmk_faces_idx[None], std.lmk_bary_coords[None])[0].numpy()
    return cKDTree(vsmpl_canon).query(sxlmk)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', nargs='+', required=True, help='img:lmk pairs, FIRST=front (geometry)')
    ap.add_argument('--body_video', required=True)
    ap.add_argument('--body_landmarks', default=None,
                    help='Optional (T,478,2) npy from extract_video_landmarks.py. '
                         'With --cache it is used only once for 2D calibration; '
                         'without --cache it remains the per-frame fallback driver.')
    ap.add_argument('--cache', default=None,
                    help='v8 render cache NPZ (*_front_back_cache.npz) containing verts_cam_all/K.')
    ap.add_argument('--target_gvhmr', default=None,
                    help='Target hmr4d_results.pt, required with --cache to map SMPL-X static landmarks to SMPL verts.')
    ap.add_argument('--target_frame', type=int, default=0)
    ap.add_argument('--uv_obj', default=HEAD_TEMPLATE_OBJ,
                    help='OBJ containing FLAME vt/ft. If unavailable, falls back to vertex colour.')
    ap.add_argument('--tex_size', type=int, default=1024,
                    help='UV texture atlas size. Use 2048 for sharper faces if runtime is acceptable.')
    ap.add_argument('--face_region', choices=['inner', 'face', 'head'], default='face',
                    help='inner=A: only central facial features/cheeks; face=current mask; head=B: full FLAME head.')
    ap.add_argument('--mask_dilate', type=int, default=9,
                    help='Pixels to dilate rendered face/head mask before feathering.')
    ap.add_argument('--mask_blur', type=float, default=2.5,
                    help='Gaussian sigma for compositing feather.')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cache and not a.target_gvhmr:
        raise ValueError("--target_gvhmr is required when --cache is used")
    if not a.cache and not a.body_landmarks:
        raise ValueError("provide either --cache (preferred) or --body_landmarks")
    # ---- multi-view SMIRK face bake (reuse) ----
    from smirk_face import reconstruct_front, bake_view, CORR
    from src.smirk_encoder import SmirkEncoder
    from src.FLAME.FLAME import FLAME
    from src.FLAME.lbs import vertices2landmarks
    enc = SmirkEncoder()
    ck = torch.load(SMIRK_CKPT, map_location='cpu')
    enc.load_state_dict({k.replace('smirk_encoder.', ''): v for k, v in ck.items() if 'smirk_encoder' in k})
    enc.eval(); flame = FLAME().eval()
    fip, flp = a.views[0].split(':')
    V0, Ff, lmk51 = reconstruct_front(enc, flame, fip, flp)
    cols, viss, dense_views = [], [], []
    for vv in a.views:
        ip, lp = vv.split(':')
        vcol, vis = bake_view(V0, Ff, lmk51, ip, lp); cols.append(vcol); viss.append(vis)
        dense_views.append(bake_view_dense(V0, Ff, lmk51, ip, lp))
    viss = np.stack(viss, 0); cols = np.stack(cols, 0)
    best = np.argmax(viss, 0); vcol = cols[best, np.arange(cols.shape[1])]; seen = viss.max(0) > 0
    if (~seen).any() and seen.any():
        skin = np.median(vcol[seen], 0); tr = cKDTree(V0[seen]); dist, nn = tr.query(V0[~seen])
        w = np.clip(dist / (0.5 * np.linalg.norm(V0.max(0) - V0.min(0))), 0, 1)[:, None]
        vcol[~seen] = (1 - w) * vcol[seen][nn] + w * skin
    # Select the rendered region. "inner" deliberately avoids the outer facial
    # outline to reduce stuck-on mask edges; "head" is the full FLAME head test.
    cen = lmk51.mean(0); rad = float(np.linalg.norm(lmk51 - cen, axis=1).max())
    dist_v = np.linalg.norm(V0 - cen, axis=1)
    if a.face_region == 'inner':
        keep = dist_v < 0.92 * rad
    elif a.face_region == 'head':
        keep = np.ones(len(V0), dtype=bool)
    else:
        keep = dist_v < 1.3 * rad
    Fface_idx = np.where(keep[Ff].all(1))[0]
    Fface = Ff[Fface_idx]
    vt, ft = parse_flame_obj_uv(a.uv_obj, Ff)
    tex = None
    if vt is not None:
        tex = bake_uv_atlas(V0, Ff, vt, ft, Fface_idx, dense_views, int(a.tex_size))
    else:
        print(f"[uv][WARN] cannot load FLAME UV from {a.uv_obj}; using per-vertex colours")
    # For the BODY alignment use DECA's landmark embedding (proven consistent with
    # SMPL-X's static-51 order) applied to the SMIRK verts -> avoids the tilt/distort
    # from SMIRK's own landmark table order differing from SMPL-X.
    _le = np.load(f"{DECA_DIR}/data/landmark_embedding.npy", allow_pickle=True)[()]
    _lfi = torch.as_tensor(_le['static_lmk_faces_idx']).long()
    _lbc = torch.as_tensor(_le['static_lmk_bary_coords']).float()
    flmk = vertices2landmarks(torch.from_numpy(V0)[None].float(), flame.faces_tensor,
                              _lfi[None], _lbc[None])[0].cpu().numpy()
    print(f"[smirk] face baked, {int(seen.sum())}/{len(V0)} verts real, "
          f"{len(Fface)} {a.face_region} faces")
    region_alpha = np.ones(len(V0), dtype=np.float32)
    if a.face_region == 'head':
        Fh, ch, sh = _head_frame(flmk)
        Vloc = (V0 - ch) @ Fh
        Lloc = (flmk - ch) @ Fh
        eye_y = float(Lloc[19:31, 1].mean())
        mouth_y = float(Lloc[31:51, 1].mean())
        chin_y = float(Lloc[:, 1].max())
        # Full-head rendering is useful for geometry, but hard edges around the
        # hairline, ears and neck look pasted. Keep the central face opaque and
        # fade toward original hair/neck/side silhouette.
        top = smoothstep(eye_y - 0.55 * sh, eye_y - 0.18 * sh, Vloc[:, 1])
        bottom = 1.0 - smoothstep(chin_y + 0.05 * sh, chin_y + 0.50 * sh, Vloc[:, 1])
        side = 1.0 - smoothstep(1.15 * sh, 1.55 * sh, np.abs(Vloc[:, 0]))
        region_alpha = np.clip(top * bottom * side, 0.0, 1.0).astype(np.float32)
        print(f"[blend] head alpha fade: min={region_alpha.min():.2f}, "
              f"mean={region_alpha.mean():.2f}, max={region_alpha.max():.2f}")
    # ---- per-frame alignment: prefer v8 SMPL cache; fallback to per-frame detection ----
    from smirk_face import CORR
    flmk14 = np.array([flmk[si] for _, si in CORR], np.float64)   # FLAME 3D (14 iBUG pts)
    mp_idx = [mi for mi, _ in CORR]
    body_lmk = np.load(a.body_landmarks) if a.body_landmarks else None
    if body_lmk is not None and not a.cache:
        # fill SHORT detection dropouts (<= max_gap frames) by linear interpolation.
        max_gap = 12
        det = ~np.isnan(body_lmk[:, 0, 0])
        idx = np.where(det)[0]
        if len(idx) >= 2:
            for a0, b0 in zip(idx[:-1], idx[1:]):
                if 1 < (b0 - a0) <= max_gap:
                    for t in range(a0 + 1, b0):
                        w = (t - a0) / (b0 - a0)
                        body_lmk[t] = (1 - w) * body_lmk[a0] + w * body_lmk[b0]
    cap = cv2.VideoCapture(a.body_video); W = int(cap.get(3)); H = int(cap.get(4)); fps = cap.get(5)
    T_video = int(cap.get(7)); wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    fpx = 1.5 * max(W, H); Kc = np.array([[fpx, 0, W / 2], [0, fpx, H / 2], [0, 0, 1]], np.float64)
    cache_verts = cache_K = smpl_lmk_vids = None
    uv_scale, uv_shift = 1.0, np.zeros(2, np.float64)
    if a.cache:
        C = np.load(a.cache)
        cache_verts = C["verts_cam_all"].astype(np.float64)
        cache_K = C["K"].astype(np.float64)
        cache_W = int(C["W"]) if "W" in C.files else W
        cache_H = int(C["H"]) if "H" in C.files else H
        if cache_W != W or cache_H != H:
            print(f"[cache][WARN] cache resolution {cache_W}x{cache_H} != video {W}x{H}; using video size")
        smpl_lmk_vids = build_smpl_lmk_vids(a.target_gvhmr, a.target_frame, cache_verts[0])
        T = min(T_video, cache_verts.shape[0])
        if body_lmk is not None:
            det = np.where(~np.isnan(body_lmk[:T, 0, 0]))[0]
            if len(det):
                tc = int(det[0])
                s0, R0, tt0 = sim_procrustes_svd(flmk, cache_verts[tc][smpl_lmk_vids])
                Vc0 = (s0 * (R0 @ V0.T).T) + tt0
                fl0 = (s0 * (R0 @ flmk14.T).T) + tt0
                pp = (cache_K @ fl0.T).T
                pred2d = pp[:, :2] / (pp[:, 2:3] + 1e-9)
                obs2d = body_lmk[tc][mp_idx].astype(np.float64)
                pred_c, obs_c = pred2d.mean(0), obs2d.mean(0)
                pred_s = np.linalg.norm(pred2d - pred_c, axis=1).mean() + 1e-9
                obs_s = np.linalg.norm(obs2d - obs_c, axis=1).mean() + 1e-9
                uv_scale = float(np.clip(obs_s / pred_s, 0.70, 1.35))
                uv_shift = obs_c - uv_scale * pred_c
                print(f"[cache] calibrated on frame {tc}: 2D scale={uv_scale:.3f}, shift={np.round(uv_shift,1).tolist()}")
            else:
                print("[cache][WARN] body_landmarks has no detected frame; no 2D calibration applied")
        print(f"[cache] using SMPL head pose for {T}/{T_video} frames")
    else:
        T = T_video
    print(f"[composite] {T} frames {W}x{H}")
    nfit = 0
    for t in range(T):
        ok, frame = cap.read()
        if not ok:
            break
        # v8 can output flat grey frames when the posed body is fully out of view.
        # The SMPL cache still has a head pose there; do not paste a floating face.
        frame_is_empty = float(frame.std()) < 6.0
        have_pose = False
        if frame_is_empty:
            have_pose = False
        elif cache_verts is not None:
            s, R, tt = sim_procrustes_svd(flmk, cache_verts[t][smpl_lmk_vids])
            Vp = (s * (R @ V0.T).T) + tt
            pp = (cache_K @ Vp.T).T
            uv = pp[:, :2] / (pp[:, 2:3] + 1e-9)
            if uv_scale != 1.0 or np.any(np.abs(uv_shift) > 1e-6):
                uv = uv * uv_scale + uv_shift[None, :]
            vx = np.concatenate([uv, Vp[:, 2:3]], 1)
            have_pose = True
            nfit += 1
        elif body_lmk is not None:
            L = body_lmk[t][mp_idx].astype(np.float64)
            if not np.isnan(L).any():
                ok2, rvec, tvec = cv2.solvePnP(flmk14, L, Kc, None, flags=cv2.SOLVEPNP_SQPNP)
                if ok2:
                    uv = cv2.projectPoints(V0.astype(np.float64), rvec, tvec, Kc, None)[0].reshape(-1, 2)
                    R, _ = cv2.Rodrigues(rvec)
                    camz = ((R @ V0.T).T + tvec.reshape(1, 3))[:, 2]
                    vx = np.concatenate([uv, camz[:, None]], 1)
                    have_pose = True
                    nfit += 1
        if have_pose:
            img = np.full((H, W), -1, np.int32); fz = vx[Fface].mean(1)[:, 2]; order = np.argsort(fz)[::-1]
            bary = np.zeros((H, W, 3), np.float32)
            for fi in order:
                p = vx[Fface[fi]][:, :2]
                x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int)
                x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, W - 1); y1 = min(y1, H - 1)
                if x1 < x0 or y1 < y0:
                    continue
                ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]; px = xs.ravel() + 0.5; py = ys.ravel() + 0.5
                d = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
                if abs(d) < 1e-9:
                    continue
                aa = ((p[1, 1] - p[2, 1]) * (px - p[2, 0]) + (p[2, 0] - p[1, 0]) * (py - p[2, 1])) / d
                bb = ((p[2, 1] - p[0, 1]) * (px - p[2, 0]) + (p[0, 0] - p[2, 0]) * (py - p[2, 1])) / d
                cc = 1 - aa - bb; m = (aa >= 0) & (bb >= 0) & (cc >= 0)
                if not m.any():
                    continue
                gx = xs.ravel()[m]; gy = ys.ravel()[m]; img[gy, gx] = fi; bary[gy, gx] = np.stack([aa[m], bb[m], cc[m]], 1)
            v = img >= 0
            if v.any():
                yy, xx = np.where(v)
                bbox_area = (xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1)
                max_bbox_frac = 0.60 if a.face_region == 'head' else 0.35
                max_area_frac = 0.30 if a.face_region == 'head' else 0.18
                if bbox_area > max_bbox_frac * H * W or v.sum() > max_area_frac * H * W:
                    wr.write(frame)
                    continue
                fi = img[v]
                if tex is not None and ft is not None:
                    uv_tex = np.einsum('mc,mcd->md', bary[v], vt[ft[Fface_idx[fi]]])
                    col = sample_uv_atlas(tex, uv_tex)
                else:
                    col = np.einsum('mc,mcd->md', bary[v], vcol[Fface[fi]])
                pix_alpha = np.einsum('mc,mc->m', bary[v], region_alpha[Fface[fi]])
                face = (col[:, ::-1] * 255).clip(0, 255).astype(np.uint8)
                fimg = np.zeros((H, W, 3), np.uint8); fimg[v] = face
                alpha_img = np.zeros((H, W), np.float32); alpha_img[v] = pix_alpha
                # (A) anti-ghost: DILATE the face mask so the SMIRK face fully COVERS
                # the underlying low-poly face (no rim showing), fill the grown ring
                # with nearest face colour, then feather only the outer edge.
                mask = (v.astype(np.uint8) * 255)
                if a.mask_dilate > 0:
                    ksz = int(a.mask_dilate) * 2 + 1
                    grown = cv2.dilate(mask, np.ones((ksz, ksz), np.uint8))
                else:
                    grown = mask
                ring = (grown > 0) & (mask == 0)
                if ring.any():
                    # nearest-face-pixel fill for the dilation ring (avoid black edge)
                    _, lbl = cv2.distanceTransformWithLabels((mask == 0).astype(np.uint8),
                                                             cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                    ys0, xs0 = np.where(mask > 0)
                    idx = lbl[ring] - 1
                    fimg[ring] = fimg[ys0[idx], xs0[idx]]
                    alpha_img[ring] = alpha_img[ys0[idx], xs0[idx]]
                al = cv2.GaussianBlur(alpha_img, (0, 0), float(a.mask_blur))[..., None]
                frame = (frame * (1 - al) + fimg * al).astype(np.uint8)
        wr.write(frame)
        if t == 0:
            cv2.imwrite(os.path.splitext(a.out)[0] + "_f0.png", frame)
        if t == T // 8:
            cv2.imwrite(os.path.splitext(a.out)[0] + "_a.png", frame)
    cap.release(); wr.release(); print(f"done ({nfit}/{T} frames got a face pose) ->", a.out)


if __name__ == '__main__':
    main()
