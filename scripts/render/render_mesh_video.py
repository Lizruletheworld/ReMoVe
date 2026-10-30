import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

#  Paths 
from paths import SMPL_PATH, GVHMR_ROOT, HY_ROOT, SMIRK_DIR, SMIRK_CKPT, DECA_DIR, HEAD_TEMPLATE_OBJ

DEFAULT_KIMODO = None
DEFAULT_OUT    = "outputs/mesh_video.mp4"

OUTPUT_FPS = 24


def round16(x):
    return (x // 16) * 16


def procrustes_align(src, dst):
    """torchStep7kimodoGVHMR"""
    c_s, c_d = src.mean(0), dst.mean(0)
    U, _, Vt = torch.linalg.svd((dst - c_d).T @ (src - c_s))
    R = U @ Vt
    if torch.det(R) < 0:
        U2 = U.clone(); U2[:, -1] *= -1; R = U2 @ Vt
    return R, c_d - R @ c_s


def procrustes_align_np(src, dst):
    """numpy(torch/gpu)R@src+t  dst"""
    c_s, c_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - c_d).T @ (src - c_s))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U2 = U.copy(); U2[:, -1] *= -1; R = U2 @ Vt
    t = c_d - R @ c_s
    return R.astype(np.float32), t.astype(np.float32)


def rotmat_to_rotvec_batch(R):
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    sin_a = np.sin(angle)
    axis  = np.zeros((len(R), 3), dtype=np.float32)
    valid = sin_a > 1e-8
    axis[valid] = np.stack([
        R[valid, 2, 1] - R[valid, 1, 2],
        R[valid, 0, 2] - R[valid, 2, 0],
        R[valid, 1, 0] - R[valid, 0, 1],
    ], axis=-1) / (2 * sin_a[valid, None])
    return (axis * angle[:, None]).astype(np.float32)


def software_rasterize(verts_cam, faces, H, W, K):
    proj  = (K @ verts_cam.T).T
    z_v   = proj[:, 2]
    uv    = proj[:, :2] / (z_v[:, None] + 1e-9)
    face_z     = z_v[faces].mean(axis=1)
    valid_face = np.all(z_v[faces] > 0.1, axis=1)
    order      = np.argsort(face_z)[::-1]
    img = np.zeros((H, W), dtype=np.uint16)
    for fi in order:
        if not valid_face[fi]:
            continue
        i0, i1, i2 = faces[fi]
        pts = np.round(uv[[i0, i1, i2]]).astype(np.int32).reshape(-1, 1, 2)
        xs, ys = pts[:, 0, 0], pts[:, 0, 1]
        if xs.max() < 0 or xs.min() >= W or ys.max() < 0 or ys.min() >= H:
            continue
        cv2.fillConvexPoly(img, pts, int(fi + 1))
    face_idx = img.astype(np.int32) - 1
    vy, vx = np.where(face_idx >= 0)
    fi_vals = face_idx[vy, vx]
    bary = np.zeros((H, W, 3), dtype=np.float32)
    if len(fi_vals) > 0:
        p0 = uv[faces[fi_vals, 0]]
        p1 = uv[faces[fi_vals, 1]]
        p2 = uv[faces[fi_vals, 2]]
        px = np.stack([vx.astype(np.float32) + 0.5,
                       vy.astype(np.float32) + 0.5], axis=1)
        denom = ((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1]) -
                 (p2[:, 0]-p0[:, 0])*(p1[:, 1]-p0[:, 1])) + 1e-9
        w0 = (((p1[:, 0]-px[:, 0])*(p2[:, 1]-px[:, 1]) -
                (p2[:, 0]-px[:, 0])*(p1[:, 1]-px[:, 1])) / denom)
        w1 = (((p2[:, 0]-px[:, 0])*(p0[:, 1]-px[:, 1]) -
                (p0[:, 0]-px[:, 0])*(p2[:, 1]-px[:, 1])) / denom)
        w2 = 1.0 - w0 - w1
        bary[vy, vx] = np.stack([w0, w1, w2], axis=1)
    return face_idx, bary


def face_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return n


def boundary_vertex_ids_from_faces(faces):
    """Return vertices on open mesh edges."""
    from collections import defaultdict
    edge_count = defaultdict(int)
    for f in faces:
        a, b, c = [int(x) for x in f]
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_count[(u, v)] += 1
    vids = set()
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            vids.add(u)
            vids.add(v)
    return np.array(sorted(vids), dtype=np.int64)


def boundary_components_from_faces(faces):
    """Connected components of boundary edges, returned as vertex-id arrays."""
    from collections import defaultdict, deque
    edge_count = defaultdict(int)
    for f in faces:
        a, b, c = [int(x) for x in f]
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_count[(u, v)] += 1
    adj = defaultdict(set)
    for (u, v), cnt in edge_count.items():
        if cnt == 1:
            adj[u].add(v)
            adj[v].add(u)
    seen = set()
    comps = []
    for start in list(adj.keys()):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(np.asarray(sorted(comp), dtype=np.int64))
    return comps


def ordered_ring_by_angle(verts, vids, samples=None):
    """Sort a neck-like boundary ring around the vertical y axis in x/z space."""
    vids = np.asarray(vids, dtype=np.int64)
    if len(vids) == 0:
        return vids
    pts = verts[vids]
    ctr = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 2] - ctr[2], pts[:, 0] - ctr[0])
    ring = vids[np.argsort(ang)]
    if samples is not None and len(ring) > samples:
        idx = np.linspace(0, len(ring), int(samples), endpoint=False).astype(np.int64)
        ring = ring[idx]
    return ring.astype(np.int64)


def resample_closed_ring(ring, n):
    ring = np.asarray(ring, dtype=np.int64)
    n = int(n)
    if len(ring) == 0 or n <= 0:
        return ring
    idx = np.linspace(0, len(ring), n, endpoint=False).astype(np.int64)
    idx = np.clip(idx, 0, len(ring) - 1)
    return ring[idx].astype(np.int64)


def flame_vertex_uv_from_face_uv(n_verts, faces, vt, ft):
    """Average OBJ face-corner UVs back onto FLAME vertex ids for bridge colours."""
    uv = np.zeros((int(n_verts), 2), dtype=np.float32)
    cnt = np.zeros(int(n_verts), dtype=np.float32)
    for tri, utri in zip(faces, ft):
        for v, ui in zip(tri, utri):
            uv[int(v)] += vt[int(ui)]
            cnt[int(v)] += 1.0
    good = cnt > 0
    uv[good] /= cnt[good, None]
    return uv


def barycentric_3d(p, a, b, c):
    """Barycentric coords (u,v,w) of points p projected onto triangles (a,b,c).
    p,a,b,c: (M,3). Returns (M,3), clamped to the simplex so uv stays inside the face."""
    v0 = b - a; v1 = c - a; v2 = p - a
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01 + 1e-12
    vv = (d11 * d20 - d01 * d21) / den
    ww = (d00 * d21 - d01 * d20) / den
    uu = 1.0 - vv - ww
    bc = np.stack([uu, vv, ww], axis=1)
    bc = np.clip(bc, 0.0, 1.0)
    bc /= bc.sum(1, keepdims=True) + 1e-9
    return bc


def make_ring_bridge_faces(n, layers=3):
    """Faces for a ruled surface from ring 0 (FLAME) to ring layers (SMPL)."""
    n = int(n)
    layers = int(max(1, layers))
    faces = []
    for j in range(layers):
        row0 = j * n
        row1 = (j + 1) * n
        for i in range(n):
            a = row0 + i
            b = row0 + ((i + 1) % n)
            c = row1 + i
            d = row1 + ((i + 1) % n)
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(faces, dtype=np.int32)


def bilinear_sample(img, xy):
    H, W = img.shape[:2]
    x = np.clip(xy[:, 0], 0, W - 1.001)
    y = np.clip(xy[:, 1], 0, H - 1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    c00, c10 = img[y0, x0], img[y0, x1]
    c01, c11 = img[y1, x0], img[y1, x1]
    return c00 * (1 - wx) * (1 - wy) + c10 * wx * (1 - wy) + c01 * (1 - wx) * wy + c11 * wx * wy


#  Background masking (kill white/green silhouette bleed = "") 
BG_DIST_THRESH = 30    # L1 color distance (0-255) to the corner color counted as background
PERSON_FRAC    = 0.35  # a rasterized face must cover >=this frac of person pixels to texture
                       # from a view (P1-4: 0.7 was too strict for silhouette-edge faces)
MASK_ERODE_PX  = 3     # erode the person mask inward to stay clear of the silhouette seam

#  Seam fix: bleed body colour outward + reject background-coloured samples 
SEAM_BLEED_PX  = 0     # >0: fill non-person pixels with nearest person colour up to this radius
REJECT_BG      = False # reject a sampled colour that is within BG_REJECT of the view bg colour
BG_REJECT      = 36    # L1 colour distance (0-255) to bg colour counted as background bleed


def skin_mask_bgr(img_bgr, person_mask=None):
    """Boolean skin mask via YCrCb thresholding (robust for the hanging-hand
    problem: the bare hand is skin-coloured while the clothes are dark). Limited
    to the person region if given."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 130)
    skin = cv2.medianBlur(skin.astype(np.uint8) * 255, 5) > 0
    if person_mask is not None:
        skin &= person_mask
    return skin


def bleed_person_outward(img_rgb, person_mask, radius_px):
    """Return a copy of img_rgb where every non-person pixel within `radius_px`
    of the body is replaced by its NEAREST person pixel colour. This prevents the
    near-white photo background from bleeding onto seam faces: a sample landing a
    few px off the true silhouette (from a widened body or small align error) now
    reads adjacent skin/clothing colour instead of white. The person mask itself
    is untouched (still used for view selection / rejection)."""
    if radius_px <= 0 or not person_mask.any() or person_mask.all():
        return img_rgb
    nonp = (~person_mask).astype(np.uint8)           # 1 on background, 0 on person
    dist, labels = cv2.distanceTransformWithLabels(
        nonp, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # OpenCV labels the zero-pixels (person) in row-major scan order 1..N, and
    # every pixel carries the label of its nearest zero-pixel.
    zy, zx = np.where(nonp == 0)
    out = img_rgb.copy()
    fill = (dist > 0) & (dist <= radius_px)
    idx = labels[fill] - 1
    out[fill] = img_rgb[zy[idx], zx[idx]]
    return out


def _silhouette_affine(proj_mask, person_mask, mode):
    """3x3 affine (image coords) mapping the projected-mesh silhouette onto the
    reference alpha silhouette, to absorb gross GVHMR translation/scale/rotation
    error. mode 'sim'=translation+isotropic scale, 'affine'=+rotation/shear via
    2nd moments. Returns None if degenerate."""
    py, px = np.where(proj_mask)
    ay, ax = np.where(person_mask)
    if len(px) < 50 or len(ax) < 50:
        return None
    pc = np.array([px.mean(), py.mean()], dtype=np.float64)
    ac = np.array([ax.mean(), ay.mean()], dtype=np.float64)
    if mode == 'sim':
        s = float(np.sqrt(len(ax) / max(len(px), 1)))
        M = np.array([[s, 0.0], [0.0, s]], dtype=np.float64)
    elif mode == 'affine':
        P = np.stack([px - pc[0], py - pc[1]], 0).astype(np.float64)
        Q = np.stack([ax - ac[0], ay - ac[1]], 0).astype(np.float64)
        Cp = (P @ P.T) / P.shape[1] + np.eye(2) * 1e-3
        Cq = (Q @ Q.T) / Q.shape[1] + np.eye(2) * 1e-3
        def _pw(C, p):
            w, V = np.linalg.eigh(C)
            w = np.clip(w, 1e-6, None)
            return (V * (w ** p)) @ V.T
        M = _pw(Cq, 0.5) @ _pw(Cp, -0.5)
    else:
        return None
    t = ac - M @ pc
    A = np.eye(3, dtype=np.float32)
    A[:2, :2] = M.astype(np.float32)
    A[:2, 2] = t.astype(np.float32)
    return A


def build_person_mask(img_bgr, alpha=None):
    """Boolean person mask (True=person). Prefer a real alpha matte; otherwise
    flood the uniform background inward from the image border by color."""
    H, W = img_bgr.shape[:2]
    if alpha is not None and int(alpha.min()) < 250:
        person = alpha > 128
    else:
        corners = np.stack([img_bgr[0, 0], img_bgr[0, -1],
                            img_bgr[-1, 0], img_bgr[-1, -1]]).astype(np.int32)
        bg_color = np.median(corners, axis=0)
        bg_like = (np.abs(img_bgr.astype(np.int32) - bg_color).sum(2)
                   < BG_DIST_THRESH).astype(np.uint8)
        # Keep only background components touching the image border; near-white
        # highlights inside the body do not touch the border and stay person.
        _, lab = cv2.connectedComponents(bg_like)
        border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
        border_labels = [int(l) for l in np.unique(border) if l != 0]
        outer_bg = (np.isin(lab, border_labels) if border_labels
                    else np.zeros((H, W), dtype=bool))
        person = ~outer_bg
    person = person.astype(np.uint8)
    if MASK_ERODE_PX > 0:
        k = np.ones((MASK_ERODE_PX * 2 + 1, MASK_ERODE_PX * 2 + 1), np.uint8)
        person = cv2.erode(person, k)
    return person.astype(bool)


def _uv_on_person(view, xy):
    mk = view["mask"]
    H, W = mk.shape
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    if x < 0 or x >= W or y < 0 or y >= H:
        return False
    return bool(mk[y, x])


def _scale_K_to_image(K_raw, gvhmr_path, W_img, H_img):
    """Scale K_fullimg to the supplied image size when it differs from GVHMR video."""
    src_w = src_h = None
    video_path = os.path.join(os.path.dirname(gvhmr_path), "0_input_video.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    if not src_w or not src_h:
        # K_fullimg normally has principal point near image center.
        src_w = max(1, int(round(float(K_raw[0, 2]) * 2.0)))
        src_h = max(1, int(round(float(K_raw[1, 2]) * 2.0)))

    K = K_raw.copy().astype(np.float32)
    K[0, :] *= W_img / src_w
    K[1, :] *= H_img / src_h
    return K


def load_gvhmr_mesh_frame(gvhmr_path, frame_idx, smplx_model, smplx2smpl, device,
                          betas_override=None):
    pred = torch.load(gvhmr_path, map_location="cpu", weights_only=False)
    params = {
        k: v[frame_idx:frame_idx + 1].float().to(device)
        for k, v in pred["smpl_params_incam"].items()
    }
    # Personalised body shape: swap GVHMR's near-mean betas for a shared shape
    # (e.g. from the silhouette fit) so the mesh matches the real build.
    if betas_override is not None:
        b = betas_override.to(device).float().reshape(1, -1)
        nb = params["betas"].shape[-1]
        if b.shape[-1] < nb:
            b = torch.cat([b, torch.zeros(1, nb - b.shape[-1], device=device)], -1)
        params["betas"] = b[:, :nb]
    with torch.no_grad():
        verts_smplx = smplx_model(**params).vertices[0]
        verts_smpl = torch.matmul(smplx2smpl, verts_smplx)
    return pred, verts_smpl


def build_view_gvhmr(name, gvhmr_path, img_path, faces_np, smplx_model, smplx2smpl,
                      verts_canonical, device, frame_idx=0, align_mode="none", mask_mode="alpha",
                      betas_override=None):
    """Build one independently fitted texture view using its own image, SMPL and K."""
    if img_path is None or gvhmr_path is None:
        return None
    img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot read {name} image: {img_path}")
    alpha = None
    if img_raw.ndim == 3 and img_raw.shape[2] == 4:
        alpha, img_bgr = img_raw[:, :, 3], img_raw[:, :, :3]
    elif img_raw.ndim == 2:
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_raw

    H0, W0 = img_bgr.shape[:2]
    H_img, W_img = round16(H0), round16(W0)
    if H_img <= 0 or W_img <= 0:
        raise ValueError(f"Invalid image size for {name}: {W0}x{H0}")

    # Person mask (True=person). Reference photos: use alpha matte / flood-fill.
    # Target-video frame (mask_mode="mesh"): no alpha and a real scene background,
    # so the mask is taken from the fitted mesh's own silhouette further below.
    if mask_mode == "mesh":
        person_mask = None
        if (H_img, W_img) != (H0, W0):
            img_bgr = cv2.resize(img_bgr, (W_img, H_img), interpolation=cv2.INTER_AREA)
    elif mask_mode == "none":
        # test4-style: NO person mask -> sample the projected pixel directly.
        # Keeps grazing/side faces sharp; relies on a tight GVHMR fit.
        if (H_img, W_img) != (H0, W0):
            img_bgr = cv2.resize(img_bgr, (W_img, H_img), interpolation=cv2.INTER_AREA)
        person_mask = np.ones((H_img, W_img), dtype=bool)
    else:
        person_mask = build_person_mask(img_bgr, alpha)
        if (H_img, W_img) != (H0, W0):
            img_bgr = cv2.resize(img_bgr, (W_img, H_img), interpolation=cv2.INTER_AREA)
            person_mask = cv2.resize(person_mask.astype(np.uint8), (W_img, H_img),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # View background colour, for background-bleed rejection. Use the median of
    # the four border strips (P1-2) rather than only the 4 corners: far more
    # robust to background gradients/shadows than 4 corner pixels.
    _border = 8
    _edges = np.concatenate([
        img_rgb[:_border, :].reshape(-1, 3),
        img_rgb[-_border:, :].reshape(-1, 3),
        img_rgb[:, :_border].reshape(-1, 3),
        img_rgb[:, -_border:].reshape(-1, 3),
    ], axis=0)
    bg_rgb = np.median(_edges, axis=0).astype(np.float32)

    # Seam fix: bleed body colour into the background so near-silhouette samples
    # read skin/clothing instead of white. Keep the ORIGINAL for QA/reference.
    img_sample = bleed_person_outward(img_rgb, person_mask, SEAM_BLEED_PX)

    # Skin mask (for locating the bare hand robustly vs the GVHMR 3D-arm offset).
    skin = skin_mask_bgr(img_bgr, person_mask if person_mask is not None else None)

    pred, verts_own_t = load_gvhmr_mesh_frame(
        gvhmr_path, frame_idx, smplx_model, smplx2smpl, device, betas_override=betas_override)
    verts_own = verts_own_t.cpu().numpy()
    K_raw = pred["K_fullimg"][0].float().numpy()
    K_own = _scale_K_to_image(K_raw, gvhmr_path, W_img, H_img)

    F = faces_np.shape[0]

    # Initial projection + rasterization (source-camera visibility).
    face_idx_view, _ = software_rasterize(verts_own, faces_np, H_img, W_img, K_own)

    # Target-video frame: the mesh IS this frame's fit, so its rasterized
    # silhouette is the person mask (eroded to stay off the background edge).
    if mask_mode == "mesh":
        _m = (face_idx_view >= 0).astype(np.uint8)
        if MASK_ERODE_PX > 0:
            _k = np.ones((MASK_ERODE_PX * 2 + 1, MASK_ERODE_PX * 2 + 1), np.uint8)
            _m = cv2.erode(_m, _k)
        person_mask = _m.astype(bool)

    # 2D silhouette registration: re-align the projected mesh onto the reference
    # alpha silhouette, absorbing gross GVHMR translation/scale/rotation error.
    # Skipped for the target frame (already perfectly aligned by construction).
    align_applied = "none"
    if align_mode in ("sim", "affine") and mask_mode == "alpha" and person_mask is not None:
        A = _silhouette_affine(face_idx_view >= 0, person_mask, align_mode)
        if A is not None:
            K_own = (A @ K_own).astype(np.float32)
            face_idx_view, _ = software_rasterize(verts_own, faces_np, H_img, W_img, K_own)
            align_applied = align_mode

    # A face is a usable texture source from this view only if the bulk of its
    # rasterized footprint lands on the person, not the background.
    vy, vx = np.where(face_idx_view >= 0)
    fids = face_idx_view[vy, vx]
    on_person = person_mask[vy, vx].astype(np.float64)
    tot = np.bincount(fids, minlength=F)
    onp = np.bincount(fids, weights=on_person, minlength=F)
    frac = np.divide(onp, tot, out=np.zeros_like(onp), where=tot > 0)
    visible_faces = np.where((tot > 0) & (frac >= PERSON_FRAC))[0].astype(np.int64)
    n_geo = int((tot > 0).sum())

    proj = (K_own @ verts_own.T).T
    uv_all = (proj[:, :2] / (proj[:, 2:3] + 1e-9)).astype(np.float32)

    # Per-face source-camera depth (mean vertex z) for the occlusion test.
    face_depth = verts_own[faces_np].mean(axis=1)[:, 2].astype(np.float32)

    R_align, t_align = procrustes_align_np(verts_own, verts_canonical)
    aligned = (R_align @ verts_own.T).T + t_align
    err_m = float(np.linalg.norm(aligned - verts_canonical, axis=1).mean())
    cam_dir = (R_align @ np.array([0, 0, -1], dtype=np.float32)).astype(np.float32)

    print(f"  [view:{name}] frame={frame_idx}, K_fx={K_own[0,0]:.0f}px, "
          f"image={W_img}x{H_img}, align={align_applied}, visible faces={len(visible_faces)} "
          f"(of {n_geo} geo, {n_geo-len(visible_faces)} dropped on background), "
          f"canonical alignment error={err_m*100:.1f}cm")
    if name != "target" and err_m > 0.15:
        print(f"  [view:{name}][WARN] alignment error >15cm; image pose/body shape may "
              "not be sufficiently consistent with the target first frame")

    return {
        "name": name,
        "image": img_sample,       # bled image used for colour sampling
        "image_raw": img_rgb,      # original, for reference/QA
        "bg_rgb": bg_rgb,
        "skin": skin,              # skin-colour mask (bare hand locator)
        "mask": person_mask,
        "uv_all": uv_all,
        "visible_faces": visible_faces,
        "cam_dir": cam_dir,
        "alignment_error_m": err_m,
        "src_faceid": face_idx_view.astype(np.int32),
        "face_depth": face_depth,
    }

def color_match_views(views, primary_idx, mode='mean'):
    """P1-3: align every non-primary view's colour statistics to the primary
    (target-frame) view, removing the lighting/white-balance seam between the
    AI-generated front/back images and the real target frame. Stats are taken
    only inside each view's person mask. mode='mean' shifts the mean;
    mode='meanstd' does a Reinhard mean+std transfer."""
    if mode == 'none' or primary_idx is None or views[primary_idx] is None:
        return
    pv = views[primary_idx]
    pm = pv.get('mask')
    ref_img = pv['image']
    ref_pixels = ref_img[pm] if (pm is not None and pm.any()) else ref_img.reshape(-1, 3)
    ref_mean = ref_pixels.mean(axis=0)
    ref_std = ref_pixels.std(axis=0) + 1e-6
    for vi, v in enumerate(views):
        if v is None or vi == primary_idx:
            continue
        sm = v.get('mask')
        src_pixels = v['image'][sm] if (sm is not None and sm.any()) else v['image'].reshape(-1, 3)
        src_mean = src_pixels.mean(axis=0)
        src_std = src_pixels.std(axis=0) + 1e-6
        if mode == 'mean':
            shift = (ref_mean - src_mean)[None, None, :]
            v['image'] = np.clip(v['image'] + shift, 0.0, 1.0).astype(np.float32)
            v['image_raw'] = np.clip(v['image_raw'] + shift, 0.0, 1.0).astype(np.float32)
        elif mode == 'meanstd':
            def _xfer(img):
                out = (img - src_mean[None, None, :]) / src_std[None, None, :] \
                    * ref_std[None, None, :] + ref_mean[None, None, :]
                return np.clip(out, 0.0, 1.0).astype(np.float32)
            v['image'] = _xfer(v['image'])
            v['image_raw'] = _xfer(v['image_raw'])
        print(f'  [color_match] view {v["name"]:6s}: mean shift '
              f'{np.round((ref_mean - src_mean) * 255, 1)}')


def smooth_vertex_colors_laplacian(vcolors, faces_np, iterations=3,
                                   boundary_only=False, boundary_faces=None):
    """P1-1: Laplacian smoothing of fallback vertex colours to soften the hard
    colour jump at view boundaries. boundary_only=True restricts smoothing to the
    vertices of boundary_faces (the seam/hidden region)."""
    if iterations <= 0:
        return vcolors
    V = vcolors.shape[0]
    from collections import defaultdict
    adj = defaultdict(set)
    for f in faces_np:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    if boundary_only and boundary_faces is not None and len(boundary_faces) > 0:
        target_verts = np.unique(faces_np[boundary_faces].reshape(-1))
    else:
        target_verts = np.arange(V)
    # Precompute neighbour lists for the target vertices only (vectorised mean).
    nbrs = {int(v): np.fromiter(adj[int(v)], dtype=np.int64) for v in target_verts}
    out = vcolors.copy()
    for _ in range(iterations):
        new = out.copy()
        for v in target_verts:
            nb = nbrs[int(v)]
            if nb.size:
                new[v] = out[nb].mean(axis=0) * 0.5 + out[v] * 0.5
        out = new
    return out.astype(np.float32)


def select_best_view_per_face(faces_np, normals_canonical, views,
                              primary_idx=None, primary_min_dot=0.40):
    F = faces_np.shape[0]
    best_view_idx = np.full(F, -1, dtype=np.int32)
    best_score = np.full(F, -1e9, dtype=np.float32)
    # supplementary views (e.g. front/back): pick per face by best facing angle
    for vi, view in enumerate(views):
        if view is None or vi == primary_idx:
            continue
        scores = normals_canonical @ view["cam_dir"]
        visible_mask = np.zeros(F, dtype=bool)
        visible_mask[view["visible_faces"]] = True
        better = visible_mask & (scores > best_score)
        best_view_idx[better] = vi
        best_score[better] = scores[better]
    # PRIMARY view (target-video frame) OVERRIDES wherever it can see a
    # sufficiently front-facing face: its accurate texture wins; the others
    # only fill what the target frame cannot see (back, steep sides).
    if primary_idx is not None and views[primary_idx] is not None:
        pv = views[primary_idx]
        scores = normals_canonical @ pv["cam_dir"]
        vis = np.zeros(F, dtype=bool)
        vis[pv["visible_faces"]] = True
        take = vis & (scores > primary_min_dot)
        best_view_idx[take] = primary_idx
    return best_view_idx


def downgrade_grazing_faces(view_idx, normals, views, grazing_dot):
    """P0-4: any face whose chosen view sees it at a grazing angle
    (normal.cam_dir < grazing_dot) is set to -1 (hidden), so it uses the smooth
    fallback vertex colour instead of a badly-stretched grazing texture sample 
    the root cause of the side 'ribbon' seams. Returns (view_idx, n_downgraded)."""
    if grazing_dot <= 0:
        return view_idx, 0
    n0 = int((view_idx < 0).sum())
    for vi, view in enumerate(views):
        if view is None:
            continue
        dot_scores = normals @ view["cam_dir"]
        grazing = (view_idx == vi) & (dot_scores < grazing_dot)
        view_idx[grazing] = -1
    return view_idx, int((view_idx < 0).sum()) - n0


def compute_blend_weights(normals, views, active_view_ids, faces_np, head_y,
                          blend_power, head_frac):
    """Per-face per-view soft-blend weights = max(0, normal.cam_dir)^power, gated
    by each view's visibility. Head faces (top head_frac by height) collapse to a
    single best view to avoid ghosting the face. Shared by the canonical precompute
    and the per-frame path (P1-5)."""
    F = faces_np.shape[0]
    W = np.zeros((F, len(views)), dtype=np.float32)
    for vi in active_view_ids:
        sc = np.clip(normals @ views[vi]["cam_dir"], 0.0, None) ** blend_power
        vis = np.zeros(F, dtype=bool)
        vis[views[vi]["visible_faces"]] = True
        W[vis, vi] = sc[vis]
    if head_frac > 0:
        head = head_y > np.quantile(head_y, 1.0 - head_frac)   # P0-1: top-by-height
        hf = np.where(head & (W.sum(axis=1) > 0))[0]
        if len(hf):
            best = np.argmax(W[hf], axis=1)
            W[hf] = 0.0
            W[hf, best] = 1.0
    return W


def load_matting_mask(matting_path, frame_idx, H, W):
    """P2-1: read an alpha/matte mask (bool, True=person) from a video or image."""
    if matting_path is None or not os.path.exists(matting_path):
        return None
    if matting_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        frame = cv2.imread(matting_path, cv2.IMREAD_GRAYSCALE)
    else:
        cap = cv2.VideoCapture(matting_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if frame is None:
        return None
    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
    return frame > 128


def bake_fallback_vertex_colors(verts_ref, faces_np, best_view_idx, views,
                                arm_face=None, exempt_face=None):
    _seed_exempt = exempt_face if exempt_face is not None else arm_face
    V = verts_ref.shape[0]
    vcolor = np.full((V, 3), -1.0, dtype=np.float32)

    def _in_arm(view, xy):
        am = view.get("arm_mask") if isinstance(view, dict) else None
        if am is None:
            return False
        H, W = am.shape
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        return 0 <= x < W and 0 <= y < H and bool(am[y, x])

    non_hidden_faces = np.where(best_view_idx >= 0)[0]
    for fi in non_hidden_faces:
        vi = best_view_idx[fi]
        view = views[vi]
        face_is_arm = bool(_seed_exempt[fi]) if _seed_exempt is not None else False
        for vidx in faces_np[fi]:
            if vcolor[vidx, 0] < 0:
                xy = view["uv_all"][vidx]
                if not _uv_on_person(view, xy):
                    continue  # never seed fallback color from the background
                if not face_is_arm and _in_arm(view, xy):
                    continue  # never seed a non-arm vertex from the hand footprint
                vcolor[vidx] = bilinear_sample(view["image"], xy[None, :])[0]

    unset = np.where(vcolor[:, 0] < 0)[0]
    k = 0
    if len(unset) > 0:
        from scipy.spatial import cKDTree
        valid_idx = np.where(vcolor[:, 0] >= 0)[0]
        # A non-arm vertex under the hand must NOT inherit skin colour from arm
        # donors: give non-arm queries only non-arm donors (so the clothing region
        # fills with nearby clothing, not the hand's skin tone).
        arm_vert = np.zeros(V, dtype=bool)   # P0-2: correct init (was always all-False)
        if arm_face is not None:
            arm_vert[faces_np[arm_face].reshape(-1)] = True
        k = int(min(8, len(valid_idx)))
        tree_all = cKDTree(verts_ref[valid_idx])
        _, nn = tree_all.query(verts_ref[unset], k=k)
        nn = nn.reshape(len(unset), k)
        vcolor[unset] = vcolor[valid_idx[nn]].mean(axis=1)
        if arm_face is not None:
            nonarm_valid = valid_idx[~arm_vert[valid_idx]]
            q = unset[~arm_vert[unset]]
            if len(q) > 0 and len(nonarm_valid) > 0:
                kk = int(min(8, len(nonarm_valid)))
                tree_na = cKDTree(verts_ref[nonarm_valid])
                _, nn2 = tree_na.query(verts_ref[q], k=kk)
                nn2 = nn2.reshape(len(q), kk)
                vcolor[q] = vcolor[nonarm_valid[nn2]].mean(axis=1)
    print(f"  [fallback] {len(unset)}  {k} ")
    return vcolor.astype(np.float32)



def extract_video_frame(video_path, frame_idx, save_path):
    """Extract one video frame, save it as PNG, and return the saved path."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Cannot find target video: {video_path}")
    if frame_idx < 0:
        raise ValueError(f"target_frame must be >= 0, got {frame_idx}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open target video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0 and frame_idx >= frame_count:
        cap.release()
        raise IndexError(
            f"target_frame={frame_idx} is outside video range [0, {frame_count - 1}]: "
            f"{video_path}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(
            f"Failed to decode frame {frame_idx} from target video: {video_path}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if not cv2.imwrite(save_path, frame_bgr):
        raise RuntimeError(f"Failed to save extracted target frame: {save_path}")
    print(f"  extracted target frame {frame_idx}: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kimodo', required=True,
                        help='motion npz: Kimodo output (step 6) or a GVHMR fit converted by gvhmr_to_kimodo_gt.py')
    parser.add_argument('--target_gvhmr', required=True,
                        help=' hmr4d_results.pt')
    parser.add_argument('--front_img', required=True)
    parser.add_argument('--front_gvhmr', required=True)
    parser.add_argument('--back_img', default=None,
                        help='Required unless --skin_body is set: --skin_body overwrites the '
                             'whole body with a flat shaded colour, so the back photo\'s texture '
                             'contribution is discarded anyway and no back reference is needed.')
    parser.add_argument('--back_gvhmr', default=None)
    parser.add_argument('--target_frame', type=int, default=0)
    parser.add_argument('--front_frame', type=int, default=0)
    parser.add_argument('--back_frame', type=int, default=0)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fps', type=float, default=OUTPUT_FPS)
    parser.add_argument('--output_scale', type=float, default=1.0,
                        help='Scale the target-camera output resolution while scaling K accordingly.')
    parser.add_argument('--output_width', type=int, default=None,
                        help='Optional explicit output width; rounded down to a multiple of 16.')
    parser.add_argument('--output_height', type=int, default=None,
                        help='Optional explicit output height; rounded down to a multiple of 16.')
    parser.add_argument('--align', choices=['none', 'sim', 'affine'], default='none',
                        help='2D silhouette registration of each reference view onto its alpha matte.')
    parser.add_argument('--occlusion', action='store_true',
                        help='Reject texture samples occluded by a nearer body part in the source view.')
    parser.add_argument('--z_tol', type=float, default=0.05,
                        help='Depth tolerance (m) for the occlusion test. (P1-8: 0.05 '
                             'reduces false occlusion rejects; was 0.02.)')
    parser.add_argument('--use_target_texture', action='store_true', default=True,
                        help='Use the target-video first frame as PRIMARY texture (front/back supplement).')
    parser.add_argument('--no_target_texture', dest='use_target_texture', action='store_false',
                        help='Disable target-frame texturing (front/back only, = v5align behaviour).')
    parser.add_argument('--target_min_dot', type=float, default=0.40,
                        help='Min (face normal . target camera dir) for a face to take the '
                             'target-frame texture. (P0-3: 0.40 stops the frontal target '
                             'texture from being force-stretched onto side faces; was 0.15.)')
    parser.add_argument('--no_kimodo', action='store_true',
                        help='Skip the Kimodo motion; render only the target first-frame (canonical) mesh textured.')
    parser.add_argument('--turntable', type=int, default=0,
                        help='With --no_kimodo, render N frames orbiting the canonical mesh about its vertical axis.')
    parser.add_argument('--target_img', default=None,
                        help='Explicit PRIMARY texture image (e.g. the ORIGINAL high-res video frame). Overrides extracting the frame from the downscaled 0_input_video.mp4.')
    parser.add_argument('--mesh_mask_all', action='store_true',
                        help='Use each view mesh silhouette as the person mask (robust when reference images have cluttered/real backgrounds and no alpha).')
    parser.add_argument('--no_mask', action='store_true',
                        help='test4-style: disable person masking entirely and sample projected pixels directly (sharpest sides; use when GVHMR fits are tight).')
    parser.add_argument('--blend', action='store_true',
                        help='Multi-view weighted blend (feather seams) instead of hard per-face view selection.')
    parser.add_argument('--blend_power', type=float, default=3.0,
                        help='Blend weight = max(0, normal.cam_dir)^power. Higher = sharper (closer to hard selection); lower = smoother.')
    parser.add_argument('--head_frac', type=float, default=0.12,
                        help='Top fraction (by height) treated as head: those faces use a single best view (no cross-view blend) to avoid ghosting the face. 0 disables.')
    parser.add_argument('--left_img', default=None, help='Optional left/side reference image.')
    parser.add_argument('--left_gvhmr', default=None, help='GVHMR fit for --left_img.')
    parser.add_argument('--right_img', default=None, help='Optional right/side reference image.')
    parser.add_argument('--right_gvhmr', default=None, help='GVHMR fit for --right_img.')
    parser.add_argument('--erode', type=int, default=None,
                        help='Override MASK_ERODE_PX (person-mask inward erosion, px) to suppress silhouette-edge background bleed.')
    #  Personalisation: body shape (betas) + absolute height (scale) 
    parser.add_argument('--betas_pt', default=None,
                        help='Path to a torch file with a personalised shape {"betas": (1,N)} '
                             '(e.g. from personalize_shape.py). Overrides GVHMR betas everywhere.')
    parser.add_argument('--betas', default=None,
                        help='Comma-separated betas, e.g. "0.1,1.8,..." . Overrides GVHMR betas. '
                             'Ignored if --betas_pt is given.')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Uniform body scale about its centroid (absolute size). 1.0 = unchanged.')
    parser.add_argument('--height_m', type=float, default=None,
                        help='Target standing height in metres; overrides --scale by computing '
                             's = height_m / current mesh stature.')
    #  Seam fix (white background bleed at widened-body silhouette) 
    parser.add_argument('--seam_bleed', type=int, default=8,
                        help='Fill non-person pixels within N px of the body with the nearest '
                             'body colour, so seam samples read skin/clothing not white. 0=off.')
    parser.add_argument('--reject_bg', action='store_true', default=True,
                        help='Reject sampled colours matching the view background (bleed guard).')
    parser.add_argument('--no_reject_bg', dest='reject_bg', action='store_false')
    parser.add_argument('--bg_reject', type=float, default=36.0,
                        help='L1 colour distance (0-255) to bg colour counted as background bleed.')
    parser.add_argument('--debug_source', action='store_true',
                        help='Render a per-pixel source map (which view fed each pixel) for QA.')
    #  Arm/hand de-ghosting (stop the hanging hand printing on torso/thigh) 
    parser.add_argument('--arm_deghost', action='store_true', default=True,
                        help='Forbid non-arm faces from sampling texture inside the hand/forearm '
                             'footprint of each source view (kills the hand imprint on clothes).')
    parser.add_argument('--no_arm_deghost', dest='arm_deghost', action='store_false')
    parser.add_argument('--arm_dilate', type=int, default=18,
                        help='Dilate the hand/forearm source footprint by N px to absorb the '
                             'GVHMR arm-pose offset before forbidding it for non-arm faces.')
    parser.add_argument('--arm_skin', action='store_true', default=False,
                        help='Also mask skin-coloured pixels within the arm vicinity (catches a '
                             'hand offset from the 3D footprint). Off by default: can over-mask '
                             'the waist where the forearm hangs beside the torso.')
    #  Flat hand: paint palm/finger faces one uniform skin colour 
    parser.add_argument('--flat_hand', action='store_true', default=True,
                        help='Paint hand (palm+finger) faces a single uniform skin '
                             'colour instead of per-pixel texture/fallback (kills the '
                             'smeared halo on hands).')
    parser.add_argument('--no_flat_hand', dest='flat_hand', action='store_false')
    parser.add_argument('--flat_face', action='store_true', default=False,
                        help='Paint the front of the face a flat skin colour (no low-poly '
                             'features) so a real face can be composited on top later '
                             'without double-face/ghosting. Needs --flat_hand skin colour.')
    parser.add_argument('--smirk_head_views', nargs='+', default=None,
                        help='Optional img:lmk pairs. If set, cut SMPL head faces and render '
                             'a SMIRK/FLAME head into the same v8 pass (true mesh replacement).')
    parser.add_argument('--smirk_head_tex_size', type=int, default=1024,
                        help='UV atlas size for --smirk_head_views.')
    parser.add_argument('--smirk_head_uv_obj',
                        default=HEAD_TEMPLATE_OBJ,
                        help='OBJ with FLAME vt/ft used for SMIRK head UV rendering.')
    parser.add_argument('--smirk_neck_fade', action='store_true', default=True,
                        help='Fade FLAME lower neck into the original SMPL neck/body to soften the seam.')
    parser.add_argument('--no_smirk_neck_fade', dest='smirk_neck_fade', action='store_false')
    parser.add_argument('--smirk_preserve_smpl_neck', action='store_true', default=True,
                        help='Keep the original SMPL neck/lower-jaw region and render only the upper FLAME head.')
    parser.add_argument('--no_smirk_preserve_smpl_neck', dest='smirk_preserve_smpl_neck',
                        action='store_false')
    parser.add_argument('--smirk_neck_bridge', action='store_true', default=False,
                        help='Build a collar mesh from the FLAME lower boundary to the SMPL neck cut.')
    parser.add_argument('--no_smirk_neck_bridge', dest='smirk_neck_bridge', action='store_false')
    parser.add_argument('--smirk_neck_bridge_samples', type=int, default=96,
                        help='Number of vertices per bridge ring.')
    parser.add_argument('--smirk_neck_bridge_layers', type=int, default=3,
                        help='Number of interpolated rings between FLAME and SMPL neck.')
    #  v9: FACE-TEXTURE-ONLY mode 
    parser.add_argument('--face_texture_only', action='store_true', default=False,
                        help='v9: do NOT replace head geometry. Keep the SMPL head+neck mesh '
                             'intact and only re-texture the front-face region of the SMPL mesh '
                             'with the SMIRK/FLAME face atlas (no FLAME mesh is rasterised, no head '
                             'is cut). Kills the jaw/neck seam of full-head replacement. '
                             'Requires --smirk_head_views (used to build the face atlas).')
    parser.add_argument('--face_tex_dist', type=float, default=0.03,
                        help='v9: max distance (SMPL cam units, ~metres) from an SMPL face-region '
                             'vertex to the nearest FLAME surface vertex for it to receive face '
                             'texture. Larger = texture more of the head; smaller = only the tight '
                             'front-face patch.')
    parser.add_argument('--face_tex_hair_frac', type=float, default=0.35,
                        help='v9: drop the top this-fraction of head height (hair/crown) from the '
                             'face-texture region so only the face below the hairline is retextured.')
    parser.add_argument('--face_tex_feather', type=int, default=4,
                        help='v9: width (mesh-ring smoothing iterations) of the alpha feather at the '
                             'face-texture region boundary, so the FLAME face blends into the '
                             'surrounding SMPL texture instead of a hard edge. 0 = hard edge.')
    parser.add_argument('--skin_body', action='store_true', default=False,
                        help='v9: do NOT texture the body from the photos. Paint the WHOLE SMPL '
                             'mesh a skin colour (the flat-hand skin tone / --hand_color) with '
                             'normal-based form shading, and only render the real face on top '
                             '(use with --face_texture_only). Result = a shaded skin-coloured body '
                             'mannequin + real textured face.')
    parser.add_argument('--skin_shade', type=float, default=0.4,
                        help='v9: strength of the normal-based light/dark shading on the skin body '
                             '(0 = flat uniform colour, 0.4 = gentle form shading, higher = stronger '
                             'contrast between camera-facing and grazing/silhouette faces).')
    parser.add_argument('--skin_color', default=None,
                        help='v9: override the --skin_body body colour, "R,G,B" in 0-255 (RGB). '
                             'e.g. "150,180,235" for the classic light-blue SMPL mesh look. '
                             'If unset, the colour is sampled from the face (cheeks/forehead).')
    parser.add_argument('--hand_color', default=None,
                        help='Override the flat hand colour, "R,G,B" in 0-255. '
                             'Default: auto median of skin-coloured reference pixels.')
    #  v8 seam/correspondence improvements 
    parser.add_argument('--grazing_dot', type=float, default=0.12,
                        help='P0-4: if the chosen view sees a face at a grazing angle '
                             '(normal.cam_dir < this), drop it to fallback vertex colour '
                             'instead of stretching the texture. 0=disable.')
    parser.add_argument('--person_frac', type=float, default=0.35,
                        help='P1-4: min fraction of a face footprint that must land on the '
                             'person mask to be usable from a view (0.35 recommended).')
    parser.add_argument('--smooth_seam_iters', type=int, default=5,
                        help='P1-1: Laplacian smoothing iterations on fallback vertex colours '
                             'at view boundaries (softens seams). 0=disable, 3~8 recommended.')
    parser.add_argument('--color_match', choices=['none', 'mean', 'meanstd'], default='none',
                        help='P1-3: align front/back image colour to the target frame. '
                             'mean=shift, meanstd=Reinhard transfer, none=disable. '
                             'NOTE default is none: it relies on the target person mask, '
                             'which is unreliable when the target frame has a real (non-matte) '
                             'background -> a wrong global shift brightens the fallback. Only '
                             'enable when the reference views have clean alpha/mesh masks.')
    parser.add_argument('--static_view', action='store_true', default=False,
                        help='P1-5: use canonical-frame normals for view selection in every '
                             'frame (v7 behaviour). Default: recompute per-frame view selection '
                             '(more accurate correspondence under large pose change).')
    parser.add_argument('--min_blend_wsum', type=float, default=0.05,
                        help='P1-6: in --blend mode, faces whose total view weight is below '
                             'this go to fallback instead of a 50/50 ghost blend.')
    parser.add_argument('--bleed_original', action='store_true', default=False,
                        help='P2-1: paste back original-frame pixels inside the person matte '
                             'but outside the mesh silhouette (recovers hair / loose clothing). '
                             'Only sound for --no_kimodo or well-aligned poses.')
    parser.add_argument('--bleed_original_matting', default=None,
                        help='Matting video/image for --bleed_original; falls back to '
                             'flood-fill background removal on the original frame if omitted.')
    parser.add_argument('--debug_video', action='store_true', default=False,
                        help='P2-2: write a side-by-side <out>_debug.mp4 (left=render, '
                             'right=view attribution: target=red front=green back=blue '
                             'side=cyan/magenta fallback=yellow).')
    parser.add_argument('--temporal_alpha', type=float, default=0.0,
                        help='P2-3: temporal EMA on rendered pixels (0=off, 0.3 moderate, '
                             '0.6 heavy). Reduces boundary flicker, adds motion blur.')
    args = parser.parse_args()
    if not args.skin_body and not (args.back_img and args.back_gvhmr):
        parser.error('--back_img/--back_gvhmr are required unless --skin_body is set')
    if args.erode is not None:
        global MASK_ERODE_PX
        MASK_ERODE_PX = int(args.erode)
    global SEAM_BLEED_PX, REJECT_BG, BG_REJECT, PERSON_FRAC
    SEAM_BLEED_PX = int(args.seam_bleed)
    REJECT_BG = bool(args.reject_bg)
    BG_REJECT = float(args.bg_reject)
    PERSON_FRAC = float(args.person_frac)   # P1-4
    for _name in (
        'kimodo', 'target_gvhmr', 'front_img', 'front_gvhmr',
        'back_img', 'back_gvhmr', 'out', 'device'
    ):
        _value = getattr(args, _name)
        if isinstance(_value, str):
            setattr(args, _name, _value.strip())

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sys.path.insert(0, GVHMR_ROOT)
    if HY_ROOT:
        sys.path.insert(0, HY_ROOT)

    #  Mock pytorch3d transforms required by imported project code 
    import types
    def _aa_to_mat(aa):
        angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis = aa / angle
        cos, sin = torch.cos(angle)[..., None], torch.sin(angle)[..., None]
        K = torch.zeros(*axis.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
        K[...,0,1]=-axis[...,2]; K[...,0,2]=axis[...,1]
        K[...,1,0]=axis[...,2];  K[...,1,2]=-axis[...,0]
        K[...,2,0]=-axis[...,1]; K[...,2,1]=axis[...,0]
        I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(*K.shape)
        return I + sin*K + (1-cos)*(K@K)
    def _r6d_to_mat(d6):
        a1, a2 = d6[..., :3], d6[..., 3:6]
        b1 = torch.nn.functional.normalize(a1, dim=-1)
        b2 = torch.nn.functional.normalize(
            a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
        return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)
    _p3d_t = types.ModuleType('pytorch3d.transforms')
    _p3d_t.axis_angle_to_matrix = _aa_to_mat
    _p3d_t.rotation_6d_to_matrix = _r6d_to_mat
    _p3d = types.ModuleType('pytorch3d'); _p3d.transforms = _p3d_t
    sys.modules.setdefault('pytorch3d', _p3d)
    sys.modules['pytorch3d.transforms'] = _p3d_t

    print('[1] Loading body models ...')
    from smplx import SMPL
    from hmr4d.utils.smplx_utils import make_smplx
    smpl_ref = SMPL(model_path=SMPL_PATH, batch_size=1, create_transl=False).eval().to(device)
    faces_np = smpl_ref.faces.astype(np.int32)
    # Label hand/forearm faces via SMPL skinning weights (dominant joint per vertex).
    # SMPL joints: 18/19 elbow, 20/21 wrist, 22/23 hand -> forearm+hand region.
    _wj = smpl_ref.lbs_weights.argmax(dim=1).cpu().numpy()
    ARM_JOINTS = [18, 19, 20, 21, 22, 23]          # forearm + wrist + hand
    arm_vert = np.isin(_wj, ARM_JOINTS)
    arm_face = arm_vert[faces_np].any(axis=1)       # face touching forearm/hand
    # Faces EXEMPT from arm-rejection: arm itself + upper arm/shoulder (16,17).
    # Upper-arm skin sampling the forearm zone is the same tone, so no visible
    # imprint; exempting it avoids a fallback band across the elbow.
    exempt_face = np.isin(_wj, [16, 17] + ARM_JOINTS)[faces_np].any(axis=1)
    HAND_JOINTS = [22, 23]                          # palm + fingers only
    hand_face = np.isin(_wj, HAND_JOINTS)[faces_np].any(axis=1)
    print(f'[arm] {int(arm_face.sum())} forearm/hand faces, '
          f'{int(exempt_face.sum())} exempt (arm+upper-arm), '
          f'{int(hand_face.sum())} hand (palm+finger) faces')
    smplx_model = make_smplx('supermotion').eval().to(device)
    smplx2smpl = torch.load(
        os.path.join(GVHMR_ROOT, 'hmr4d/utils/body_model/smplx2smpl_sparse.pt'),
        map_location=device)

    #  Resolve personalised betas (shape). Priority: --betas_pt > --betas > GVHMR 
    betas_p = None
    if args.betas_pt:
        _bp = torch.load(args.betas_pt.strip(), map_location='cpu', weights_only=False)
        betas_p = (_bp['betas'] if isinstance(_bp, dict) and 'betas' in _bp else _bp).float()
        print(f'[shape] using personalised betas from {args.betas_pt}: '
              f'{[round(float(x),3) for x in betas_p.reshape(-1)]}')
    elif args.betas:
        betas_p = torch.tensor([float(x) for x in args.betas.split(',')]).float().reshape(1, -1)
        print(f'[shape] using --betas: {[round(float(x),3) for x in betas_p.reshape(-1)]}')

    # Target selected-frame GVHMR mesh replaces REF_PKL as canonical geometry.
    # The target frame is NOT used as a texture image.
    print('[2] Loading target canonical mesh and camera ...')
    gvhmr_pred, verts_gvhmr_0 = load_gvhmr_mesh_frame(
        args.target_gvhmr, args.target_frame,
        smplx_model, smplx2smpl, device, betas_override=betas_p)
    verts_canonical = verts_gvhmr_0.cpu().numpy()

    # Output resolution comes from the target GVHMR input video when available;
    # otherwise infer it from the principal point in K_fullimg.
    target_video_path = os.path.join(
        os.path.dirname(os.path.abspath(args.target_gvhmr)),
        '0_input_video.mp4')
    if os.path.exists(target_video_path):
        cap = cv2.VideoCapture(target_video_path)
        OUT_W = round16(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        OUT_H = round16(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    else:
        K0 = gvhmr_pred['K_fullimg'][0].float().numpy()
        OUT_W = round16(int(round(float(K0[0, 2]) * 2.0)))
        OUT_H = round16(int(round(float(K0[1, 2]) * 2.0)))

    if OUT_W <= 0 or OUT_H <= 0:
        raise ValueError(
            f'Cannot determine target output resolution from {args.target_gvhmr}')

    base_OUT_W, base_OUT_H = OUT_W, OUT_H
    if args.output_width is not None or args.output_height is not None:
        if args.output_width is not None and args.output_width <= 0:
            raise ValueError('--output_width must be positive')
        if args.output_height is not None and args.output_height <= 0:
            raise ValueError('--output_height must be positive')
        if args.output_width is not None and args.output_height is not None:
            OUT_W = round16(args.output_width)
            OUT_H = round16(args.output_height)
        elif args.output_width is not None:
            OUT_W = round16(args.output_width)
            OUT_H = round16(int(round(base_OUT_H * (OUT_W / base_OUT_W))))
        else:
            OUT_H = round16(args.output_height)
            OUT_W = round16(int(round(base_OUT_W * (OUT_H / base_OUT_H))))
    elif args.output_scale != 1.0:
        if args.output_scale <= 0:
            raise ValueError('--output_scale must be positive')
        OUT_W = round16(int(round(base_OUT_W * args.output_scale)))
        OUT_H = round16(int(round(base_OUT_H * args.output_scale)))

    if OUT_W <= 0 or OUT_H <= 0:
        raise ValueError(
            f'Invalid scaled output resolution: {OUT_W}x{OUT_H}')

    K_raw = gvhmr_pred['K_fullimg'][0].float().numpy()
    K_np = _scale_K_to_image(K_raw, args.target_gvhmr, OUT_W, OUT_H)
    K_tgt = torch.from_numpy(K_np).float().to(device)
    print(f'  base output resolution: {base_OUT_W}x{base_OUT_H}')
    print(f'  output resolution: {OUT_W}x{OUT_H}')

    print('[3] Building texture views (target frame primary + front/back) ...')
    views = []
    primary_idx = None
    if args.use_target_texture:
        _tmm = 'none' if args.no_mask else ('mesh' if args.mesh_mask_all else 'alpha')
        if args.target_img:
            views.append(build_view_gvhmr('target', args.target_gvhmr, args.target_img, faces_np,
                                          smplx_model, smplx2smpl, verts_canonical, device,
                                          args.target_frame, 'none', _tmm, betas_override=betas_p))
            primary_idx = 0
            print('  [target] using explicit --target_img as PRIMARY texture: %s' % args.target_img)
        else:
            tgt_video = os.path.join(os.path.dirname(os.path.abspath(args.target_gvhmr)),
                                     '0_input_video.mp4')
            if os.path.exists(tgt_video):
                tgt_frame_png = os.path.splitext(os.path.abspath(args.out))[0] + '_targetframe.png'
                extract_video_frame(tgt_video, args.target_frame, tgt_frame_png)
                views.append(build_view_gvhmr('target', args.target_gvhmr, tgt_frame_png, faces_np,
                                              smplx_model, smplx2smpl, verts_canonical, device,
                                              args.target_frame, 'none', _tmm, betas_override=betas_p))
                primary_idx = 0
                print('  [target] using target-video frame %d as PRIMARY texture source' % args.target_frame)
            else:
                print('  [target] %s not found -> front/back only' % tgt_video)
    _rmm = 'none' if args.no_mask else ('mesh' if args.mesh_mask_all else 'alpha')
    views.append(build_view_gvhmr('front', args.front_gvhmr, args.front_img, faces_np,
                     smplx_model, smplx2smpl, verts_canonical, device,
                     args.front_frame, args.align, _rmm, betas_override=betas_p))
    if args.back_img and args.back_gvhmr:
        views.append(build_view_gvhmr('back', args.back_gvhmr, args.back_img, faces_np,
                         smplx_model, smplx2smpl, verts_canonical, device,
                         args.back_frame, args.align, _rmm, betas_override=betas_p))
    if args.left_img and args.left_gvhmr:
        views.append(build_view_gvhmr('left', args.left_gvhmr, args.left_img, faces_np,
                         smplx_model, smplx2smpl, verts_canonical, device,
                         0, args.align, _rmm, betas_override=betas_p))
    if args.right_img and args.right_gvhmr:
        views.append(build_view_gvhmr('right', args.right_gvhmr, args.right_img, faces_np,
                         smplx_model, smplx2smpl, verts_canonical, device,
                         0, args.align, _rmm, betas_override=betas_p))

    #  P2-4: sanity-check we have at least one usable view 
    _active = [i for i, v in enumerate(views) if v is not None]
    if len(_active) == 0:
        raise RuntimeError('No valid texture views could be built. '
                           'Check image paths and GVHMR files.')
    print(f'  : {[views[i]["name"] for i in _active]}')

    #  P1-3: cross-view colour calibration (align front/back to the target frame) 
    if args.color_match != 'none' and primary_idx is not None:
        color_match_views(views, primary_idx, mode=args.color_match)
        print(f'  [color_match]  target frame (mode={args.color_match})')

    #  Per-view hand/forearm footprint (dilated) for arm de-ghosting 
    if args.arm_deghost:
        for v in views:
            if v is None:
                continue
            sf = v["src_faceid"]
            core = (sf >= 0) & arm_face[np.clip(sf, 0, len(arm_face) - 1)]  # true 3D hand/forearm
            if args.arm_dilate > 0:
                k = np.ones((args.arm_dilate * 2 + 1, args.arm_dilate * 2 + 1), np.uint8)
                zone = cv2.dilate(core.astype(np.uint8), k).astype(bool)     # plausible vicinity
            else:
                zone = core
            # The bare hand is skin-coloured; the real 2D hand may sit off the 3D
            # footprint (GVHMR arm error). Optionally mask skin within the vicinity
            # too (--arm_skin) -> catches the offset hand, but can over-mask the
            # waist where the forearm hangs beside the torso, so it's opt-in.
            skin = v.get("skin")
            if args.arm_skin and skin is not None:
                am = core | (zone & skin)
            else:
                am = zone
            v["arm_mask"] = am
            print(f'  [arm] view {v["name"]:6s}: mask {int(am.sum())} px '
                  f'(core {int(core.sum())}, dilate {args.arm_dilate}, +skin)')

    #  Uniform hand colour: median of skin-coloured reference pixels 
    hand_rgb = None
    if args.flat_hand:
        if args.hand_color:
            hand_rgb = (np.array([float(x) for x in args.hand_color.split(',')],
                                 dtype=np.float32)[:3] / 255.0)
        else:
            # P1-7: prefer the primary (target-frame) view's skin tone; only fall
            # back to other views if the target has no skin pixels. Mixing all
            # views' skin can bias the hand colour toward the AI images' lighting.
            _sk_primary, _sk_others = [], []
            for vi, _v in enumerate(views):
                if _v is None:
                    continue
                _m = _v.get('skin')
                if _m is None or not _m.any():
                    continue
                pixels = _v['image_raw'][_m]   # raw (un-bled) skin pixels
                (_sk_primary if vi == primary_idx else _sk_others).append(pixels)
            if _sk_primary:
                hand_rgb = np.median(np.concatenate(_sk_primary, 0), axis=0).astype(np.float32)
            elif _sk_others:
                hand_rgb = np.median(np.concatenate(_sk_others, 0), axis=0).astype(np.float32)
        if hand_rgb is not None:
            print(f'[hand] flat hand colour (RGB 0-1) = '
                  f'{[round(float(x), 3) for x in hand_rgb]}')
        else:
            print('[hand] no skin pixels found in any view -> flat-hand skipped')

    print('[3.5] Selecting per-face view (target primary, front/back supplement) ...')
    normals_canonical = face_normals(verts_canonical, faces_np)
    best_view_idx = select_best_view_per_face(faces_np, normals_canonical, views,
                                              primary_idx=primary_idx,
                                              primary_min_dot=args.target_min_dot)
    n_hidden_faces = int((best_view_idx < 0).sum())
    print(f'  {n_hidden_faces}/{len(faces_np)} faces invisible in all three views')
    # P0-4: drop grazing faces to fallback (kills stretched side-seam ribbons).
    best_view_idx, n_graze = downgrade_grazing_faces(
        best_view_idx, normals_canonical, views, args.grazing_dot)
    if n_graze:
        print(f'  {n_graze}  face  fallback (dot < {args.grazing_dot})')
    # Faces to paint a flat skin colour: hand (palm+finger) always, plus any
    # forearm/wrist face that is invisible in every view (its fallback colour is
    # the dark nearest-neighbour smear that bleeds off the clothes onto the hand).
    hidden_face = best_view_idx < 0
    paint_face = hand_face | (arm_face & hidden_face)
    # --flat_face: paint the FRONT of the skull (joint 15, camera-facing) a flat skin
    # colour so no low-poly facial features are drawn -> a clean base for a real face
    # composited on top later (no double-face / ghosting).
    if getattr(args, 'flat_face', False):
        _camd = (np.asarray(views[primary_idx]["cam_dir"], np.float32)
                 if primary_idx is not None else np.array([0, 0, -1], np.float32))
        _skull_v = np.isin(_wj, [15])
        _skull_f = _skull_v[faces_np].any(axis=1)
        # restrict to BELOW the hairline: keep the top ~40% of the head height as hair.
        _sy = verts_canonical[_skull_v][:, 1]
        _ytop, _ybot = _sy.min(), _sy.max()            # camera y-down: min=crown, max=chin
        _fcy = verts_canonical[faces_np].mean(axis=1)[:, 1]
        _below = _fcy > (_ytop + 0.40 * (_ybot - _ytop))
        skull_front = _skull_f & (normals_canonical @ _camd > 0.0) & _below
        paint_face = paint_face | skull_front
        print(f'  [flat_face] painting {int(skull_front.sum())} front-face faces flat skin (hair kept)')
    print(f'  [hand] flat-paint {int(paint_face.sum())} faces '
          f'(hand {int(hand_face.sum())} + hidden-arm {int((arm_face & hidden_face).sum())})')
    fallback_vertex_colors = bake_fallback_vertex_colors(
        verts_canonical, faces_np, best_view_idx, views,
        arm_face=arm_face if args.arm_deghost else None,
        exempt_face=exempt_face if args.arm_deghost else None)

    # P1-1: Laplacian-smooth fallback colours over the seam/hidden region so the
    # KDTree nearest-neighbour fill does not create a hard colour edge at view
    # boundaries. Restricted to seam faces to keep the rest crisp.
    if args.smooth_seam_iters > 0:
        seam_faces = np.where(best_view_idx < 0)[0]
        if len(seam_faces):
            fallback_vertex_colors = smooth_vertex_colors_laplacian(
                fallback_vertex_colors, faces_np,
                iterations=args.smooth_seam_iters,
                boundary_only=True, boundary_faces=seam_faces)
            print(f'  [seam] Laplacian  {args.smooth_seam_iters} , '
                  f' {len(seam_faces)}  face ')

    betas_tgt = gvhmr_pred['smpl_params_incam']['betas'][
        args.target_frame:args.target_frame + 1].float().to(device)
    if betas_p is not None:
        _b = betas_p.to(device).float().reshape(1, -1)
        nb = betas_tgt.shape[-1]
        if _b.shape[-1] < nb:
            _b = torch.cat([_b, torch.zeros(1, nb - _b.shape[-1], device=device)], -1)
        betas_tgt = _b[:, :nb]

    #  Absolute-height scale (applied to RENDER vertices only, about centroid;
    #    texture/normal logic keeps the unscaled canonical mesh). 
    def _apply_scale(verts_t, s):
        if abs(s - 1.0) < 1e-6:
            return verts_t
        c = verts_t.reshape(-1, 3).mean(0, keepdim=True)
        return (verts_t - c) * s + c
    stature_m = float(verts_gvhmr_0[:, 1].max() - verts_gvhmr_0[:, 1].min())
    if args.height_m is not None:
        if args.height_m <= 0:
            raise ValueError('--height_m must be positive')
        scale_s = args.height_m / max(stature_m, 1e-6)
        print(f'[scale] current stature={stature_m:.3f} m -> target {args.height_m:.3f} m, '
              f's={scale_s:.4f}')
    else:
        scale_s = args.scale
        if abs(scale_s - 1.0) > 1e-6:
            print(f'[scale] current stature={stature_m:.3f} m, --scale={scale_s:.4f} '
                  f'-> {stature_m*scale_s:.3f} m')
    verts_gvhmr_0 = _apply_scale(verts_gvhmr_0, scale_s)

    #  6/7. Pose sequence to render 
    if args.no_kimodo:
        v0 = verts_gvhmr_0
        if args.turntable and args.turntable > 0:
            N = args.turntable
            print("[6/7] No Kimodo: turntable of %d frames about the body vertical axis" % N)
            c = v0.mean(0, keepdim=True)
            Xnp = (v0 - c).detach().cpu().numpy()
            _, _, vt = np.linalg.svd(Xnp, full_matrices=False)
            ax = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
            Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=np.float64)
            frames = []
            for i in range(N):
                th = 2.0 * np.pi * i / N
                Rm = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
                R = torch.tensor(Rm, device=v0.device, dtype=v0.dtype)
                frames.append((v0 - c) @ R.T + c)
            verts_cam_all = torch.stack(frames)
            T = N
        else:
            print("[6/7] No Kimodo: rendering the target first-frame (canonical) mesh only")
            verts_cam_all = v0.unsqueeze(0)
            T = 1
    else:
        print("[6] Loading Kimodo ...")
        kfr            = np.load(args.kimodo)
        local_rot_mats = kfr["local_rot_mats"]
        root_positions  = kfr["root_positions"]
        T = local_rot_mats.shape[0]
        print("  %d frames" % T)
        global_orient_aa = rotmat_to_rotvec_batch(local_rot_mats[:, 0])
        body_pose_aa     = rotmat_to_rotvec_batch(
            local_rot_mats[:, 1:].reshape(-1, 3, 3)).reshape(T, 63)
        with torch.no_grad():
            out_k = smplx_model(
                global_orient=torch.from_numpy(global_orient_aa).to(device),
                body_pose    =torch.from_numpy(body_pose_aa).to(device),
                betas        =betas_tgt.expand(T, -1),
                transl       =torch.from_numpy(root_positions).float().to(device))
            verts_kimodo_all = torch.stack([
                torch.matmul(smplx2smpl, out_k.vertices[i]) for i in range(T)])
        print("[7] Aligning ...")
        R_align, t_align = procrustes_align(verts_kimodo_all[0], verts_gvhmr_0)
        verts_cam_all = (R_align @ verts_kimodo_all.permute(0, 2, 1)).permute(0, 2, 1) + t_align
        # Match the personalised absolute height on the moving sequence too,
        # scaling about the frame-0 centroid so the body stays anchored.
        if abs(scale_s - 1.0) > 1e-6:
            c0 = verts_cam_all[0].mean(0, keepdim=True)
            verts_cam_all = (verts_cam_all - c0) * scale_s + c0
        err0 = (verts_cam_all[0] - verts_gvhmr_0).norm(dim=1).mean().item()
        print("  f0 alignment error: %.1f cm" % (err0 * 100))

    #  7.5  
    kimodo_name = "firstframe" if args.no_kimodo else os.path.splitext(os.path.basename(args.kimodo))[0]
    cache_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                               f"{kimodo_name}_front_back_cache.npz")
    np.savez_compressed(
        cache_path,
        verts_cam_all=verts_cam_all.cpu().numpy().astype(np.float32),
        faces=faces_np.astype(np.int32),
        K=K_tgt.cpu().numpy().astype(np.float32),
        W=OUT_W, H=OUT_H,
        best_view_idx=best_view_idx.astype(np.int32),
        fallback_vertex_colors=fallback_vertex_colors.astype(np.float32),
    )
    print(f"  [cache] saved to {cache_path}")

    #  7.7 Optional TRUE head replacement: remove SMPL head faces, then render
    #     a SMIRK/FLAME head in the same raster pass. This avoids the old
    #     SMPL-image -> FLAME-image -> 2D-alpha pipeline where two heads overlap.
    smirk_head = None
    head_cut_face = None
    face_tex = None           # v9: face-texture-only payload (built below when --face_texture_only)
    if args.smirk_head_views:
        print("[7.7] Building SMIRK/FLAME replacement head ...")
        _cwd = os.getcwd()
        os.chdir(SMIRK_DIR)
        try:
            from smirk_face import reconstruct_front, bake_view
            from src.smirk_encoder import SmirkEncoder
            from src.FLAME.FLAME import FLAME as SmirkFLAME
            from src.FLAME.lbs import vertices2landmarks as smirk_vertices2landmarks
            from smirk_bodyface import (
                bake_view_dense, bake_uv_atlas, parse_flame_obj_uv,
                sample_uv_atlas, sim_procrustes_svd, build_smpl_lmk_vids,
            )
            enc = SmirkEncoder()
            ck = torch.load(SMIRK_CKPT, map_location='cpu')
            enc.load_state_dict({k.replace('smirk_encoder.', ''): v for k, v in ck.items()
                                 if 'smirk_encoder' in k})
            enc.eval()
            smirk_flame = SmirkFLAME().eval()
            fip, flp = args.smirk_head_views[0].split(':')
            Vh, Fh, lmk51 = reconstruct_front(enc, smirk_flame, fip, flp)
            dense_views = []
            cols, viss = [], []
            for vv in args.smirk_head_views:
                ip, lp = vv.split(':')
                vcol, vis = bake_view(Vh, Fh, lmk51, ip, lp)
                cols.append(vcol); viss.append(vis)
                dense_views.append(bake_view_dense(Vh, Fh, lmk51, ip, lp))
            vt_uv, ft_uv = parse_flame_obj_uv(args.smirk_head_uv_obj, Fh)
            if vt_uv is None:
                raise RuntimeError(f"Cannot load FLAME UV from {args.smirk_head_uv_obj}")
            head_face_idx = np.arange(len(Fh), dtype=np.int64)
            head_tex = bake_uv_atlas(Vh, Fh, vt_uv, ft_uv, head_face_idx,
                                     dense_views, int(args.smirk_head_tex_size))
            _le = np.load(f"{DECA_DIR}/data/landmark_embedding.npy", allow_pickle=True)[()]
            _lfi = torch.as_tensor(_le['static_lmk_faces_idx']).long()
            _lbc = torch.as_tensor(_le['static_lmk_bary_coords']).float()
            flmk = smirk_vertices2landmarks(torch.from_numpy(Vh)[None].float(),
                                            smirk_flame.faces_tensor,
                                            _lfi[None], _lbc[None])[0].cpu().numpy()
        finally:
            os.chdir(_cwd)
        smpl_lmk_vids = build_smpl_lmk_vids(args.target_gvhmr, args.target_frame,
                                            verts_cam_all[0].detach().cpu().numpy())
        head_joint_vert = np.isin(_wj, [15])
        head_joint_face = head_joint_vert[faces_np].any(axis=1)
        if args.smirk_preserve_smpl_neck:
            v0 = verts_cam_all[0].detach().cpu().numpy()
            hy = v0[head_joint_vert, 1]
            h_top, h_bot = float(hy.min()), float(hy.max())
            smpl_cut_y = h_top + 0.92 * (h_bot - h_top)
            face_y = v0[faces_np].mean(axis=1)[:, 1]
            head_cut_face = head_joint_face & (face_y < smpl_cut_y)

            flame_face_y = Vh[Fh].mean(axis=1)[:, 1]
            flame_cut_y = float(np.percentile(flame_face_y, 4.0))
            keep_flame_face = flame_face_y >= flame_cut_y
            if int(keep_flame_face.sum()) >= 100:
                Fh = Fh[keep_flame_face]
                ft_uv = ft_uv[keep_flame_face]
            print(f"  [smirk_head] preserve SMPL neck: cut_y={smpl_cut_y:.4f}, "
                  f"cut {int(head_cut_face.sum())}/{int(head_joint_face.sum())} SMPL head faces; "
                  f"keep {len(Fh)} FLAME upper-head faces")
        else:
            head_cut_face = head_joint_face
        head_alpha = np.ones(len(Vh), dtype=np.float32)
        if args.smirk_preserve_smpl_neck:
            fade0 = float(np.percentile(Vh[:, 1], 4.0))
            fade1 = float(np.percentile(Vh[:, 1], 14.0))
            fade_t = np.clip((Vh[:, 1] - fade0) / (fade1 - fade0 + 1e-9), 0.0, 1.0)
            head_alpha = (fade_t * fade_t * (3.0 - 2.0 * fade_t)).astype(np.float32)
            print(f"  [smirk_head] preserve-neck lower FLAME fade: y {fade0:.4f} -> {fade1:.4f}")
        elif args.smirk_neck_fade and not args.smirk_neck_bridge:
            mouth_y = float(flmk[31:51, 1].mean())
            chin_y = float(flmk[:, 1].max())
            fade0 = mouth_y + 0.55 * (chin_y - mouth_y)
            fade1 = chin_y + 0.70 * (chin_y - mouth_y)
            fade_t = np.clip((Vh[:, 1] - fade0) / (fade1 - fade0 + 1e-9), 0.0, 1.0)
            head_alpha = (1.0 - (fade_t * fade_t * (3.0 - 2.0 * fade_t))).astype(np.float32)
            print(f"  [smirk_head] neck fade enabled: y {fade0:.4f} -> {fade1:.4f}")
        neck_bridge = None
        if args.smirk_neck_bridge:
            n_bridge = int(max(12, args.smirk_neck_bridge_samples))
            flame_comps = boundary_components_from_faces(Fh)
            if flame_comps:
                lower = max(flame_comps, key=lambda c: float(Vh[c, 1].mean()))
            else:
                flame_bv = np.arange(len(Vh), dtype=np.int64)
                yb = Vh[flame_bv, 1]
                lower = flame_bv[np.argsort(yb)[-min(len(flame_bv), n_bridge):]]
            flame_ring = resample_closed_ring(
                ordered_ring_by_angle(Vh, lower), n_bridge)

            body_comps = boundary_components_from_faces(faces_np[~head_cut_face])
            if body_comps:
                body_open = max(body_comps, key=len)
            else:
                body_open = boundary_vertex_ids_from_faces(faces_np[~head_cut_face])
            body_pts0 = verts_cam_all[0].detach().cpu().numpy()[body_open]
            if len(body_open) > 16:
                ctr_xz = np.median(body_pts0[:, [0, 2]], axis=0)
                d_xz = np.linalg.norm(body_pts0[:, [0, 2]] - ctr_xz[None, :], axis=1)
                keep = d_xz <= np.percentile(d_xz, 60)
                if int(keep.sum()) >= 12:
                    body_open = body_open[keep]
            body_ring = resample_closed_ring(
                ordered_ring_by_angle(verts_cam_all[0].detach().cpu().numpy(), body_open),
                n_bridge)
            if len(flame_ring) >= 3 and len(body_ring) >= 3:
                flame_uv = flame_vertex_uv_from_face_uv(len(Vh), Fh, vt_uv, ft_uv)
                bridge_faces = make_ring_bridge_faces(n_bridge, args.smirk_neck_bridge_layers)
                neck_bridge = {
                    "flame_ring": flame_ring.astype(np.int64),
                    "body_ring": body_ring.astype(np.int64),
                    "flame_uv": flame_uv.astype(np.float32),
                    "faces": bridge_faces.astype(np.int32),
                    "layers": int(max(1, args.smirk_neck_bridge_layers)),
                }
                print(f"  [smirk_head] neck bridge enabled: "
                      f"{len(flame_ring)} FLAME ring -> {len(body_ring)} SMPL ring, "
                      f"{len(bridge_faces)} faces")
            else:
                print("  [smirk_head][WARN] neck bridge skipped: boundary ring too small")
        smirk_head = {
            "V": Vh.astype(np.float64), "F": Fh.astype(np.int32),
            "vt": vt_uv.astype(np.float32), "ft": ft_uv.astype(np.int32),
            "tex": head_tex.astype(np.float32), "flmk": flmk.astype(np.float64),
            "alpha": head_alpha.astype(np.float32),
            "neck_bridge": neck_bridge,
            "smpl_head_face": head_joint_face.astype(bool),
            "smpl_neck_face": (head_joint_face & (~head_cut_face)).astype(bool),
            "smpl_lmk_vids": smpl_lmk_vids.astype(np.int64),
        }
        print(f"  [smirk_head] cut {int(head_cut_face.sum())} SMPL head faces; "
              f"render {len(Fh)} FLAME head faces in unified pass")

        #  v9: FACE-TEXTURE-ONLY. Keep the whole SMPL head+neck mesh; only re-texture
        #     the front-face region with the FLAME face atlas (no FLAME mesh rendered,
        #     no head cut). Builds a one-time SMPL-vertex -> FLAME-UV map at the ref frame.
        if args.face_texture_only:
            from scipy.spatial import cKDTree
            v_ref = verts_cam_all[args.target_frame].detach().cpu().numpy()
            # region mask: SMPL head faces that are camera-facing, below the hairline, AND
            # sit close to the FLAME surface (so only the real front-face patch is retextured).
            s_a, R_a, t_a = sim_procrustes_svd(flmk, v_ref[smpl_lmk_vids])
            Vh_ref = (s_a * (R_a @ Vh.T).T) + t_a
            head_vids = np.where(head_joint_vert)[0]
            dist, _ = cKDTree(Vh_ref).query(v_ref[head_vids])
            near_vert = np.zeros(len(v_ref), dtype=bool)
            near_vert[head_vids] = dist < float(args.face_tex_dist)
            _camd = (np.asarray(views[primary_idx]["cam_dir"], np.float32)
                     if primary_idx is not None else np.array([0, 0, -1], np.float32))
            _hyc = verts_canonical[head_joint_vert][:, 1]
            _ytop, _ybot = float(_hyc.min()), float(_hyc.max())
            _fcy = verts_canonical[faces_np].mean(axis=1)[:, 1]
            _below = _fcy > (_ytop + float(args.face_tex_hair_frac) * (_ybot - _ytop))
            _front = (normals_canonical @ _camd) > 0.0
            _face_near = near_vert[faces_np].all(axis=1)
            face_region_face = head_joint_face & _face_near & _front & _below
            # Feather alpha over the region boundary (per SMPL vertex): 1 deep inside the
            # face patch, fading to 0 at the edge, so the FLAME texture blends into the
            # surrounding SMPL texture instead of a hard line.
            _nv = len(v_ref)
            _vt_reg = np.zeros(_nv, np.float32); _vt_head = np.zeros(_nv, np.float32)
            _headf = np.where(head_joint_face)[0]
            for _fi in _headf:
                _isreg = face_region_face[_fi]
                for _v in faces_np[_fi]:
                    _vt_head[_v] += 1.0
                    if _isreg:
                        _vt_reg[_v] += 1.0
            alpha_v = np.zeros(_nv, np.float32)
            _has = _vt_head > 0
            alpha_v[_has] = _vt_reg[_has] / _vt_head[_has]
            if int(args.face_tex_feather) > 0:
                _adj = {}
                for _fi in _headf:
                    _a, _b, _c = faces_np[_fi]
                    for _u, _w in ((_a, _b), (_a, _c), (_b, _c)):
                        _adj.setdefault(_u, set()).add(_w)
                        _adj.setdefault(_w, set()).add(_u)
                for _ in range(int(args.face_tex_feather)):
                    _new = alpha_v.copy()
                    for _u, _nb in _adj.items():
                        if _nb:
                            _new[_u] = 0.5 * alpha_v[_u] + 0.5 * float(np.mean(
                                [alpha_v[_k] for _k in _nb]))
                    alpha_v = _new
            # Apply the blend on an EXPANDED region: any camera-facing head face the feather
            # reaches (alpha>0), so the transition fades on BOTH sides of the tight boundary.
            _face_alpha_max = alpha_v[faces_np].max(axis=1)
            blend_region_face = (head_joint_face & _front & _below & (_face_alpha_max > 0.02))
            # Per-PIXEL sampling payload: each face pixel is back-projected to the nearest
            # FLAME triangle (barycentric) and reads the atlas there -> sharp, correctly
            # registered texture, not limited by SMPL's sparse face vertices.
            face_tex = {
                "tex": head_tex.astype(np.float32),
                "region": blend_region_face.astype(bool),
                "sampler": sample_uv_atlas,
                "flmk": flmk.astype(np.float64),
                "smpl_lmk_vids": smpl_lmk_vids.astype(np.int64),
                "Vh": Vh.astype(np.float64),
                "Fh": Fh.astype(np.int64),
                "vt": vt_uv.astype(np.float32),
                "ft": ft_uv.astype(np.int64),
                "tree": cKDTree(Vh[Fh].mean(axis=1)),   # FLAME face centroids (local space)
                "alpha_v": alpha_v.astype(np.float32),  # per-SMPL-vertex feather weight
            }
            # Skin colour for --skin_body: sample the FRONT PHOTO at well-lit cheek/forehead
            # MediaPipe landmarks -> the person's ACTUAL face skin tone (per-person, not a
            # brown atlas-average dragged down by hair/shadow/neck). Falls back to the atlas
            # mid-tone median if the front image/landmarks are unavailable.
            face_tex["skin_rgb"] = None
            if getattr(args, 'skin_color', None):
                # explicit colour (e.g. blue SMPL-mesh look) -> skip face sampling
                face_tex["skin_rgb"] = (np.array([float(x) for x in args.skin_color.split(',')],
                                                 dtype=np.float32)[:3] / 255.0)   # RGB 0-1
            else:
                # Match the RENDERED FACE exactly: sample the atlas at the SAME face-region faces
                # that render on the head (centroid -> FLAME-local -> nearest FLAME face -> atlas),
                # so the body colour == the tone the face actually renders (not the whole-atlas
                # average, and not the brighter raw photo). Drop beard/eye-dark & specular, median.
                try:
                    _frc = np.where(face_region_face)[0]
                    if len(_frc) > 20:
                        _cent = v_ref[faces_np[_frc]].mean(axis=1)            # SMPL face centroids
                        _loc = ((_cent - t_a) @ R_a) / s_a                    # -> FLAME local
                        _, _nf = cKDTree(Vh[Fh].mean(axis=1)).query(_loc)     # nearest FLAME face
                        _uvc = vt_uv[ft_uv[_nf]].mean(axis=1)                 # per-face UV centroid
                        _cols = sample_uv_atlas(head_tex, _uvc)              # atlas colour per face
                        _l = _cols.mean(1)
                        _keep = (_l > np.percentile(_l, 20)) & (_l < np.percentile(_l, 97))
                        face_tex["skin_rgb"] = np.median(_cols[_keep], axis=0).astype(np.float32)
                except Exception as _e:
                    print(f"  [skin_body][WARN] face-region atlas sampling failed ({_e})")
            if face_tex["skin_rgb"] is None:
                _lum = head_tex.reshape(-1, 3).mean(axis=1)
                _flat = head_tex.reshape(-1, 3)
                _sel = (_lum > 0.35) & (_lum < 0.92)
                if int(_sel.sum()) > 200:
                    face_tex["skin_rgb"] = np.median(_flat[_sel], axis=0).astype(np.float32)
            if face_tex["skin_rgb"] is not None:
                print(f"  [skin_body] skin colour = "
                      f"{[round(float(x), 3) for x in face_tex['skin_rgb']]}")
            print(f"  [face_texture_only] retexture core={int(face_region_face.sum())} "
                  f"blend={int(blend_region_face.sum())} SMPL faces (per-pixel FLAME projection, "
                  f"feather={args.face_tex_feather}, dist<{args.face_tex_dist}, "
                  f"hair_frac={args.face_tex_hair_frac}); full SMPL head/neck geometry, "
                  f"no FLAME mesh / no head cut")
            smirk_head = None      # skip the FLAME raster pass and the neck flat-paint
            head_cut_face = None   # keep every SMPL face

    if args.face_texture_only and face_tex is None:
        print("  [face_texture_only][WARN] ignored: --face_texture_only needs --smirk_head_views "
              "(the face atlas source). Running as plain v8.")

    #  8. Render 
    print("[8] Rendering %d frames -> %s ..." % (T, args.out))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise RuntimeError(f'OpenCV VideoWriter failed to open output video: {args.out}')
    # mask=(mesh)=
    # mask face_idx>=0ground truth
    # ""
    # ""
    mask_path = os.path.splitext(args.out)[0] + "_mask.mp4"
    mask_writer = cv2.VideoWriter(mask_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  args.fps, (OUT_W, OUT_H))
    if not mask_writer.isOpened():
        writer.release()
        raise RuntimeError(f'OpenCV VideoWriter failed to open mask video: {mask_path}')
    verts_all_np = verts_cam_all.cpu().numpy()
    K_np         = K_tgt.cpu().numpy()
    BG_COLOR     = 0.55
    active_view_ids = [i for i, v in enumerate(views) if v is not None]

    # P2-2: optional side-by-side view-attribution debug video.
    dbg_writer = None
    if args.debug_video:
        dbg_path = os.path.splitext(args.out)[0] + '_debug.mp4'
        dbg_writer = cv2.VideoWriter(dbg_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (OUT_W * 2, OUT_H))
        if not dbg_writer.isOpened():
            print('[WARN] cannot open debug video writer; --debug_video skipped.')
            dbg_writer = None
        else:
            print(f'  [debug] writing view attribution video -> {dbg_path}')
    VIEW_COLORS = {'target': (0.90, 0.15, 0.15), 'front': (0.15, 0.85, 0.15),
                   'back': (0.20, 0.30, 0.95), 'left': (0.10, 0.85, 0.90),
                   'right': (0.95, 0.25, 0.90)}

    # Per-face per-view blend weights on the CANONICAL mesh. Reused directly when
    # --static_view; otherwise recomputed each frame from the posed normals (P1-5).
    fcy_canonical = verts_canonical[faces_np].mean(axis=1)[:, 1]
    Wface = np.zeros((faces_np.shape[0], len(views)), dtype=np.float32)
    if args.blend:
        Wface = compute_blend_weights(normals_canonical, views, active_view_ids,
                                      faces_np, fcy_canonical, args.blend_power,
                                      args.head_frac)

    ema_canvas = None   # P2-3 temporal EMA cache
    for t_idx in tqdm(range(T)):
        vt = verts_all_np[t_idx]
        face_idx_r, bary_r = software_rasterize(vt, faces_np, OUT_H, OUT_W, K_np)

        # P1-5: recompute per-face view selection from the CURRENT frame normals so
        # correspondence stays correct under large pose change (turn/side-on). The
        # canonical selection is only reused with --static_view (v7 behaviour).
        if args.static_view:
            view_idx_frame = best_view_idx
            Wface_frame = Wface
        else:
            normals_frame = face_normals(vt, faces_np)
            view_idx_frame = select_best_view_per_face(
                faces_np, normals_frame, views, primary_idx=primary_idx,
                primary_min_dot=args.target_min_dot)
            view_idx_frame, _ = downgrade_grazing_faces(
                view_idx_frame, normals_frame, views, args.grazing_dot)
            if args.blend:
                fcy_frame = vt[faces_np].mean(axis=1)[:, 1]
                Wface_frame = compute_blend_weights(
                    normals_frame, views, active_view_ids, faces_np, fcy_frame,
                    args.blend_power, args.head_frac)
            else:
                Wface_frame = Wface

        canvas = np.full((OUT_H, OUT_W, 3), BG_COLOR, dtype=np.float32)
        valid = face_idx_r >= 0
        if head_cut_face is not None:
            valid = valid & (~head_cut_face[np.clip(face_idx_r, 0, len(head_cut_face) - 1)])
        zbuf = np.full((OUT_H, OUT_W), np.inf, dtype=np.float32)
        if valid.any():
            fi_valid = face_idx_r[valid]
            bary_valid = bary_r[valid]
            zbuf[valid] = np.einsum('mc,mc->m',
                                    bary_valid,
                                    vt[faces_np[fi_valid]][:, :, 2]).astype(np.float32)
            out_colors = np.zeros((fi_valid.shape[0], 3), dtype=np.float32)

            if args.blend:
                acc = np.zeros((fi_valid.shape[0], 3), dtype=np.float32)
                wsum = np.zeros((fi_valid.shape[0],), dtype=np.float32)
                for vi in active_view_ids:
                    w = Wface_frame[fi_valid, vi]
                    m = w > 0
                    if not m.any():
                        continue
                    fi_m = fi_valid[m]; bary_m = bary_valid[m]
                    tri_verts = faces_np[fi_m]
                    uv3 = views[vi]["uv_all"][tri_verts]
                    sample_xy = np.einsum('mc,mcd->md', bary_m, uv3)
                    colors = bilinear_sample(views[vi]["image"], sample_xy)
                    mk = views[vi]["mask"]; Hm, Wm = mk.shape
                    sx = np.clip(np.round(sample_xy[:, 0]).astype(np.int32), 0, Wm - 1)
                    sy = np.clip(np.round(sample_xy[:, 1]).astype(np.int32), 0, Hm - 1)
                    good = mk[sy, sx]
                    if args.occlusion:
                        src_fid = views[vi]["src_faceid"][sy, sx]
                        fdepth = views[vi]["face_depth"]
                        occ = (src_fid >= 0) & (
                            fdepth[fi_m] - fdepth[np.clip(src_fid, 0, fdepth.shape[0] - 1)]
                            > args.z_tol)
                        good = good & (~occ)
                    wm = w[m] * good.astype(np.float32)
                    acc[m] += wm[:, None] * colors
                    wsum[m] += wm
                have = wsum > args.min_blend_wsum   # P1-6: avoid 50/50 ghost blends
                out_colors[have] = acc[have] / wsum[have, None]
                if (~have).any():
                    tv = faces_np[fi_valid[~have]]
                    out_colors[~have] = np.einsum('mc,mcd->md', bary_valid[~have],
                                                  fallback_vertex_colors[tv])
            else:
                view_of_pixel = view_idx_frame[fi_valid]
                dbg_view = np.full(fi_valid.shape[0], -1, dtype=np.int32)  # source view id
                dbg_rej = np.zeros(fi_valid.shape[0], dtype=bool)          # sent to fallback
                for vi in active_view_ids:
                    m = view_of_pixel == vi
                    if not m.any():
                        continue
                    fi_m = fi_valid[m]
                    bary_m = bary_valid[m]
                    tri_verts = faces_np[fi_m]
                    uv3 = views[vi]["uv_all"][tri_verts]
                    sample_xy = np.einsum('mc,mcd->md', bary_m, uv3)
                    colors = bilinear_sample(views[vi]["image"], sample_xy)
                    mk = views[vi]["mask"]
                    Hm, Wm = mk.shape
                    sx = np.clip(np.round(sample_xy[:, 0]).astype(np.int32), 0, Wm - 1)
                    sy = np.clip(np.round(sample_xy[:, 1]).astype(np.int32), 0, Hm - 1)
                    reject = ~mk[sy, sx]
                    if args.occlusion:
                        src_fid = views[vi]["src_faceid"][sy, sx]
                        fdepth = views[vi]["face_depth"]
                        occ = (src_fid >= 0) & (
                            fdepth[fi_m] - fdepth[np.clip(src_fid, 0, fdepth.shape[0] - 1)]
                            > args.z_tol)
                        reject = reject | occ
                    # Background-bleed guard: a sampled colour that matches the
                    # view's background (near-white photo) is rejected even if the
                    # geometric mask passed (widened body / align slack at the seam).
                    if REJECT_BG:
                        bg = views[vi]["bg_rgb"]
                        bgd = np.abs(colors - bg[None, :]).sum(axis=1) * 255.0
                        reject = reject | (bgd < BG_REJECT)
                    # Arm de-ghosting: a NON-arm face (torso/thigh) must not sample
                    # inside the hand/forearm footprint of this view, else the
                    # hanging hand prints onto the clothes.
                    if args.arm_deghost and "arm_mask" in views[vi]:
                        in_arm = views[vi]["arm_mask"][sy, sx]
                        reject = reject | (in_arm & (~exempt_face[fi_m]))
                    if reject.any():
                        colors[reject] = np.einsum('mc,mcd->md', bary_m[reject],
                                                   fallback_vertex_colors[tri_verts[reject]])
                    out_colors[m] = colors
                    dbg_view[m] = vi
                    dbg_rej[m] = reject
                hidden_m = view_of_pixel < 0
                if hidden_m.any():
                    tri_verts = faces_np[fi_valid[hidden_m]]
                    vcolors_tri = fallback_vertex_colors[tri_verts]
                    out_colors[hidden_m] = np.einsum('mc,mcd->md', bary_valid[hidden_m], vcolors_tri)

                #  source-attribution debug: recolour by which view fed the pixel 
                if getattr(args, 'debug_source', False):
                    palette = {"target": (0.9, 0.1, 0.1), "front": (0.1, 0.85, 0.1),
                               "back": (0.15, 0.3, 0.95), "left": (0.1, 0.85, 0.9),
                               "right": (0.95, 0.2, 0.9)}
                    dbg_colors = np.zeros_like(out_colors)
                    for vi in active_view_ids:
                        dbg_colors[dbg_view == vi] = palette.get(views[vi]["name"], (0.5, 0.5, 0.5))
                    dbg_colors[dbg_view < 0] = (1.0, 0.9, 0.0)   # hidden -> yellow
                    dbg_colors[dbg_rej] = (1.0, 1.0, 1.0)        # fallback (rejected) -> white
                    out_colors = dbg_colors

            # v9 --skin_body: ignore the photo texture entirely and paint the WHOLE mesh a
            # skin colour with normal-based form shading (camera-facing bright, silhouette
            # darker) so it keeps a 3D light/dark feel; the real face is composited on top.
            if getattr(args, 'skin_body', False) and not getattr(args, 'debug_source', False):
                # prefer the face-atlas skin colour (matches the face); fall back to hand_rgb
                _skc = None
                if face_tex is not None and face_tex.get("skin_rgb") is not None:
                    _skc = face_tex["skin_rgb"]
                elif hand_rgb is not None:
                    _skc = hand_rgb
                if _skc is not None:
                    _fn = np.cross(vt[faces_np[:, 1]] - vt[faces_np[:, 0]],
                                   vt[faces_np[:, 2]] - vt[faces_np[:, 0]])
                    _nz = np.abs(_fn[:, 2]) / (np.linalg.norm(_fn, axis=1) + 1e-9)  # facing-ness 0..1
                    _sh = float(args.skin_shade)
                    _shade = ((1.0 - _sh) + _sh * _nz)[fi_valid][:, None]
                    out_colors[:] = np.asarray(_skc, np.float32)[None, :] * _shade

            # Flat hand: overwrite palm/finger faces with one uniform skin colour
            # (skip in --debug_source; skip in --skin_body since the whole body is already
            #  skin-filled with shading -> a flat hand patch would look wrong).
            if hand_rgb is not None and not getattr(args, 'debug_source', False) \
                    and not getattr(args, 'skin_body', False):
                _hp = paint_face[fi_valid]
                if _hp.any():
                    out_colors[_hp] = hand_rgb
                if smirk_head is not None and smirk_head.get("smpl_neck_face") is not None:
                    _npf = smirk_head["smpl_neck_face"][fi_valid]
                    if _npf.any():
                        out_colors[_npf] = hand_rgb

            # v9 face-texture-only: re-texture the front-face region of the SMPL mesh with
            # the FLAME face atlas, PER PIXEL. Each face pixel -> its 3D SMPL point -> FLAME
            # local space -> nearest FLAME triangle (barycentric) -> atlas. SMPL geometry is
            # unchanged and no FLAME mesh is drawn.
            if face_tex is not None and not getattr(args, 'debug_source', False):
                _fr = face_tex["region"][fi_valid]
                if _fr.any():
                    s_h, R_h, t_h = sim_procrustes_svd(
                        face_tex["flmk"], vt[face_tex["smpl_lmk_vids"]])
                    _tri = faces_np[fi_valid[_fr]]                                  # (M,3)
                    _p3d = np.einsum('mc,mcd->md', bary_valid[_fr], vt[_tri])       # SMPL cam pts
                    _ploc = ((_p3d - t_h) @ R_h) / s_h                             # -> FLAME local
                    _, _nf = face_tex["tree"].query(_ploc)                          # nearest FLAME face
                    _Fsel = face_tex["Fh"][_nf]                                     # (M,3)
                    _bc = barycentric_3d(_ploc,
                                         face_tex["Vh"][_Fsel[:, 0]],
                                         face_tex["Vh"][_Fsel[:, 1]],
                                         face_tex["Vh"][_Fsel[:, 2]])              # (M,3)
                    _uvf = face_tex["vt"][face_tex["ft"][_nf]]                      # (M,3,2)
                    _uv = np.einsum('mc,mcd->md', _bc, _uvf)                        # (M,2)
                    _col = face_tex["sampler"](face_tex["tex"], _uv)
                    # feathered alpha blend into the surrounding SMPL texture (no hard edge)
                    _a = np.einsum('mc,mc->m', bary_valid[_fr],
                                   face_tex["alpha_v"][_tri])[:, None]              # (M,1)
                    _idx = np.where(_fr)[0]
                    out_colors[_idx] = _a * _col + (1.0 - _a) * out_colors[_idx]

            canvas[valid] = out_colors

        if smirk_head is not None and not getattr(args, 'debug_source', False):
            s_h, R_h, t_h = sim_procrustes_svd(
                smirk_head["flmk"], vt[smirk_head["smpl_lmk_vids"]])
            Vh_cam = (s_h * (R_h @ smirk_head["V"].T).T) + t_h
            proj_h = (K_np @ Vh_cam.T).T
            uv_h = proj_h[:, :2] / (proj_h[:, 2:3] + 1e-9)
            vx_h = np.concatenate([uv_h, Vh_cam[:, 2:3]], axis=1)
            Fh = smirk_head["F"]
            fz_h = vx_h[Fh].mean(1)[:, 2]
            order_h = np.argsort(fz_h)[::-1]  # far -> near
            head_idx = np.full((OUT_H, OUT_W), -1, np.int32)
            head_bary = np.zeros((OUT_H, OUT_W, 3), np.float32)
            head_z = np.full((OUT_H, OUT_W), np.inf, np.float32)
            for hfi in order_h:
                p = vx_h[Fh[hfi]][:, :2]
                x0, y0 = np.floor(p.min(0)).astype(int)
                x1, y1 = np.ceil(p.max(0)).astype(int)
                x0 = max(x0, 0); y0 = max(y0, 0)
                x1 = min(x1, OUT_W - 1); y1 = min(y1, OUT_H - 1)
                if x1 < x0 or y1 < y0:
                    continue
                ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
                px = xs.ravel() + 0.5; py = ys.ravel() + 0.5
                d = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) +
                     (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
                if abs(d) < 1e-9:
                    continue
                aa = ((p[1, 1] - p[2, 1]) * (px - p[2, 0]) +
                      (p[2, 0] - p[1, 0]) * (py - p[2, 1])) / d
                bb = ((p[2, 1] - p[0, 1]) * (px - p[2, 0]) +
                      (p[0, 0] - p[2, 0]) * (py - p[2, 1])) / d
                cc = 1 - aa - bb
                inside = (aa >= 0) & (bb >= 0) & (cc >= 0)
                if not inside.any():
                    continue
                gx = xs.ravel()[inside]; gy = ys.ravel()[inside]
                bc = np.stack([aa[inside], bb[inside], cc[inside]], axis=1).astype(np.float32)
                hz = bc @ vx_h[Fh[hfi]][:, 2]
                nearer = hz < head_z[gy, gx]
                if nearer.any():
                    gx2 = gx[nearer]; gy2 = gy[nearer]
                    head_idx[gy2, gx2] = hfi
                    head_bary[gy2, gx2] = bc[nearer]
                    head_z[gy2, gx2] = hz[nearer]
            smpl_head_px = ((face_idx_r >= 0) &
                            smirk_head["smpl_head_face"][
                                np.clip(face_idx_r, 0, len(smirk_head["smpl_head_face"]) - 1)])
            hv = (head_idx >= 0) & ((head_z <= zbuf + 1e-4) | smpl_head_px)
            if hv.any():
                hfi = head_idx[hv]
                uv_tex = np.einsum('mc,mcd->md',
                                   head_bary[hv],
                                   smirk_head["vt"][smirk_head["ft"][hfi]])
                hcol = sample_uv_atlas(smirk_head["tex"], uv_tex)
                halpha = np.einsum('mc,mc->m',
                                   head_bary[hv],
                                   smirk_head["alpha"][smirk_head["F"][hfi]])
                halpha = np.clip(halpha.astype(np.float32), 0.0, 1.0)[:, None]
                base_col = canvas[hv].copy()
                if hand_rgb is not None and smirk_head.get("smpl_neck_face") is not None:
                    missing_base = ~valid[hv]
                    if missing_base.any():
                        base_col[missing_base] = hand_rgb
                canvas[hv] = base_col * (1.0 - halpha) + hcol * halpha
                valid = valid | hv
                zbuf[hv] = head_z[hv]

            br = smirk_head.get("neck_bridge")
            if br is not None:
                fr = br["flame_ring"]
                sr = br["body_ring"]
                layers = int(br["layers"])
                Bf = br["faces"]
                top = Vh_cam[fr]
                bottom = vt[sr]
                rings = []
                cols_b = []
                top_col = sample_uv_atlas(smirk_head["tex"], br["flame_uv"][fr])
                bot_col = fallback_vertex_colors[sr]
                for j in range(layers + 1):
                    w = float(j) / float(max(1, layers))
                    rings.append((1.0 - w) * top + w * bottom)
                    cols_b.append((1.0 - w) * top_col + w * bot_col)
                Bv_cam = np.concatenate(rings, axis=0).astype(np.float64)
                Bcol = np.concatenate(cols_b, axis=0).astype(np.float32)
                proj_b = (K_np @ Bv_cam.T).T
                uv_b = proj_b[:, :2] / (proj_b[:, 2:3] + 1e-9)
                vx_b = np.concatenate([uv_b, Bv_cam[:, 2:3]], axis=1)
                fz_b = vx_b[Bf].mean(1)[:, 2]
                order_b = np.argsort(fz_b)[::-1]
                bridge_idx = np.full((OUT_H, OUT_W), -1, np.int32)
                bridge_bary = np.zeros((OUT_H, OUT_W, 3), np.float32)
                bridge_z = np.full((OUT_H, OUT_W), np.inf, np.float32)
                for bfi in order_b:
                    p = vx_b[Bf[bfi]][:, :2]
                    x0, y0 = np.floor(p.min(0)).astype(int)
                    x1, y1 = np.ceil(p.max(0)).astype(int)
                    x0 = max(x0, 0); y0 = max(y0, 0)
                    x1 = min(x1, OUT_W - 1); y1 = min(y1, OUT_H - 1)
                    if x1 < x0 or y1 < y0:
                        continue
                    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
                    px = xs.ravel() + 0.5; py = ys.ravel() + 0.5
                    d = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) +
                         (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
                    if abs(d) < 1e-9:
                        continue
                    aa = ((p[1, 1] - p[2, 1]) * (px - p[2, 0]) +
                          (p[2, 0] - p[1, 0]) * (py - p[2, 1])) / d
                    bb = ((p[2, 1] - p[0, 1]) * (px - p[2, 0]) +
                          (p[0, 0] - p[2, 0]) * (py - p[2, 1])) / d
                    cc = 1 - aa - bb
                    inside = (aa >= 0) & (bb >= 0) & (cc >= 0)
                    if not inside.any():
                        continue
                    gx = xs.ravel()[inside]; gy = ys.ravel()[inside]
                    bc = np.stack([aa[inside], bb[inside], cc[inside]], axis=1).astype(np.float32)
                    bz = bc @ vx_b[Bf[bfi]][:, 2]
                    nearer = bz < bridge_z[gy, gx]
                    if nearer.any():
                        gx2 = gx[nearer]; gy2 = gy[nearer]
                        bridge_idx[gy2, gx2] = bfi
                        bridge_bary[gy2, gx2] = bc[nearer]
                        bridge_z[gy2, gx2] = bz[nearer]
                bv = (bridge_idx >= 0) & (bridge_z <= zbuf + 1e-4)
                if bv.any():
                    bfi = bridge_idx[bv]
                    canvas[bv] = np.einsum('mc,mcd->md',
                                           bridge_bary[bv],
                                           Bcol[Bf[bfi]])
                    valid = valid | bv
                    zbuf[bv] = bridge_z[bv]

        # P2-1: paste back original-frame pixels inside the person matte but outside
        # the mesh silhouette (recovers hair / loose clothing). Pose must be aligned,
        # so this is only sound with --no_kimodo or a well-tracked target.
        if args.bleed_original and os.path.exists(target_video_path):
            _cap = cv2.VideoCapture(target_video_path)
            _cap.set(cv2.CAP_PROP_POS_FRAMES, t_idx)
            _ok, _obgr = _cap.read()
            _cap.release()
            if _ok and _obgr is not None:
                _obgr = cv2.resize(_obgr, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                _orgb = cv2.cvtColor(_obgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                _mat = load_matting_mask(args.bleed_original_matting, t_idx, OUT_H, OUT_W)
                if _mat is None:
                    _mat = build_person_mask(_obgr)
                overflow = _mat & (~valid)
                if overflow.any():
                    canvas[overflow] = _orgb[overflow]
                    valid = valid | overflow

        # P2-3: temporal EMA over the mesh region to damp view-boundary flicker.
        if args.temporal_alpha > 0:
            if ema_canvas is None:
                ema_canvas = canvas.copy()
            else:
                a = args.temporal_alpha
                ema_canvas = ema_canvas * a + canvas * (1.0 - a)
                canvas[valid] = ema_canvas[valid]

        frame = (canvas[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
        writer.write(frame)
        if t_idx == 0:
            cv2.imwrite(os.path.splitext(args.out)[0] + ".png", frame)

        # P2-2: view-attribution debug frame (side-by-side with the render).
        if dbg_writer is not None:
            dbg_canvas = np.full((OUT_H, OUT_W, 3), 0.1, dtype=np.float32)
            if valid.any():
                fi_v = face_idx_r[valid]
                vi_v = view_idx_frame[fi_v]
                dcol = np.tile(np.array([0.95, 0.85, 0.0], np.float32),
                               (fi_v.shape[0], 1))     # fallback/hidden -> yellow
                for vi in active_view_ids:
                    dcol[vi_v == vi] = VIEW_COLORS.get(views[vi]["name"], (0.5, 0.5, 0.5))
                dbg_canvas[valid] = dcol
            dbg_bgr = (dbg_canvas[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
            dbg_writer.write(np.concatenate([frame, dbg_bgr], axis=1))

        # maskvalid(mesh)=255=03
        mask_u8 = (valid.astype(np.uint8) * 255)
        mask_writer.write(cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR))

    writer.release()
    mask_writer.release()
    if dbg_writer is not None:
        dbg_writer.release()
        print("Debug -> %s" % (os.path.splitext(args.out)[0] + '_debug.mp4'))
    print("Done -> %s" % args.out)
    print("Mask -> %s" % mask_path)


if __name__ == '__main__':
    main()
