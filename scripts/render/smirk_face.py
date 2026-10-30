import os, sys, argparse
import numpy as np, cv2, torch
from skimage.transform import estimate_transform, warp
from scipy.spatial import cKDTree
from paths import SMIRK_DIR, SMIRK_CKPT
os.chdir(SMIRK_DIR); sys.path.insert(0, SMIRK_DIR)
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME


def crop_tform(kpt, scale=1.4, size=224):
    l, r = kpt[:, 0].min(), kpt[:, 0].max(); t, b = kpt[:, 1].min(), kpt[:, 1].max()
    old = (r - l + b - t) / 2; cx = r - (r - l) / 2; cy = b - (b - t) / 2; s = int(old * scale)
    src = np.array([[cx - s / 2, cy - s / 2], [cx - s / 2, cy + s / 2], [cx + s / 2, cy - s / 2]], np.float32)
    dst = np.array([[0, 0], [0, size - 1], [size - 1, 0]], np.float32)
    return estimate_transform('similarity', src, dst), size


def rot_y(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def raster(vx, F, H, W):
    img = np.full((H, W), -1, np.int32); fz = vx[F].mean(1)[:, 2]; order = np.argsort(fz)
    bary = np.zeros((H, W, 3), np.float32)
    for fi in order:
        p = vx[F[fi]][:, :2]
        x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int)
        x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, W - 1); y1 = min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]; px = xs.ravel() + 0.5; py = ys.ravel() + 0.5
        d = ((p[1, 1] - p[2, 1]) * (p[0, 0] - p[2, 0]) + (p[2, 0] - p[1, 0]) * (p[0, 1] - p[2, 1]))
        if abs(d) < 1e-9:
            continue
        a = ((p[1, 1] - p[2, 1]) * (px - p[2, 0]) + (p[2, 0] - p[1, 0]) * (py - p[2, 1])) / d
        b = ((p[2, 1] - p[0, 1]) * (px - p[2, 0]) + (p[0, 0] - p[2, 0]) * (py - p[2, 1])) / d
        c = 1 - a - b; m = (a >= 0) & (b >= 0) & (c >= 0)
        if not m.any():
            continue
        gx = xs.ravel()[m]; gy = ys.ravel()[m]; img[gy, gx] = fi; bary[gy, gx] = np.stack([a[m], b[m], c[m]], 1)
    return img, bary


def vnormals(V, F):
    nrm = np.zeros_like(V); tri = V[F]; fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(nrm, F[:, k], fn)
    return nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)


# mediapipe(478) -> iBUG static idx (=iBUG_0idx-17), same 14 corr used elsewhere
CORR = [(1, 13), (2, 16), (33, 19), (133, 22), (362, 25), (263, 28),
        (61, 31), (291, 37), (0, 34), (17, 40), (55, 4), (46, 0), (285, 5), (276, 9)]


def reconstruct_front(enc, flame, img_path, lmk_path):
    """Front view: SMIRK geometry V0 + its iBUG-static 3D landmarks (for pose fitting)."""
    kpt = np.load(lmk_path).astype(np.float32); tf, S = crop_tform(kpt)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED); bgr = img[:, :, :3] if img.ndim == 3 else img
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = warp(rgb, tf.inverse, output_shape=(S, S), preserve_range=True).astype(np.float32)
    with torch.no_grad():
        outputs = enc(torch.from_numpy(crop.transpose(2, 0, 1))[None].float())
        # Zero the global head pose -> canonical/frontal geometry (keep expression,
        # jaw, eyelids). Otherwise the front image's head tilt is baked into V0 and
        # double-counts when we later align it to the body's SMPL head.
        outputs['pose_params'] = torch.zeros_like(outputs['pose_params'])
        fo = flame.forward(outputs)
        V0 = fo['vertices'][0].cpu().numpy()
    F = flame.faces_tensor.cpu().numpy()
    from src.FLAME.lbs import vertices2landmarks
    lmk51 = vertices2landmarks(torch.from_numpy(V0)[None].float(), flame.faces_tensor,
                               flame.lmk_faces_idx[None], flame.lmk_bary_coords[None])[0].cpu().numpy()
    return V0, F, lmk51


def bake_view(V0, F, lmk51, img_path, lmk_path):
    """Fit this view's head pose (solvePnP) so V0 projects consistently, then sample
    colours. Returns (vcol, vis) aligned to V0 vertex indexing."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    alpha = img[:, :, 3] if (img.ndim == 3 and img.shape[2] == 4) else None
    bgr = img[:, :, :3] if img.ndim == 3 else img
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    person = (alpha > 128) if alpha is not None else np.ones(rgb.shape[:2], bool)
    Hf, Wf = rgb.shape[:2]
    mp = np.load(lmk_path).astype(np.float32)
    obj = np.array([lmk51[si] for _, si in CORR], np.float64)      # 3D on V0
    img2d = np.array([mp[mi] for mi, _ in CORR], np.float64)       # 2D detected
    f = 1.5 * max(Wf, Hf); Kc = np.array([[f, 0, Wf / 2], [0, f, Hf / 2], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img2d, Kc, None, flags=cv2.SOLVEPNP_SQPNP)
    R, _ = cv2.Rodrigues(rvec)
    uv = cv2.projectPoints(V0.astype(np.float64), rvec, tvec, Kc, None)[0].reshape(-1, 2)
    xi = np.clip(uv[:, 0], 0, Wf - 1).astype(int); yi = np.clip(uv[:, 1], 0, Hf - 1).astype(int)
    vcol = rgb[yi, xi].astype(np.float32)
    not_bg = ~((vcol.min(1) > 0.85) | (vcol.max(1) < 0.06))
    onp = person[yi, xi] & not_bg
    vn = vnormals(V0, F)
    vn_cam = (R @ vn.T).T                # normal in camera frame (OpenCV: +z into scene)
    vis = -vn_cam[:, 2].copy()           # >0 => faces the camera
    vis[~onp] = -1e9
    return vcol, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', nargs='+', required=True, help='img.png:lmk.npy pairs; FIRST is front (geometry)')
    ap.add_argument('--out', required=True); ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--size', type=int, default=512); ap.add_argument('--fps', type=float, default=24)
    ap.add_argument('--max_deg', type=float, default=70.0)
    ap.add_argument('--checkpoint', default=SMIRK_CKPT)
    a = ap.parse_args()
    enc = SmirkEncoder()
    ck = torch.load(a.checkpoint, map_location='cpu')
    enc.load_state_dict({k.replace('smirk_encoder.', ''): v for k, v in ck.items() if 'smirk_encoder' in k})
    enc.eval(); flame = FLAME().eval()
    fip, flp = a.views[0].split(':')
    Vgeo, F, lmk51 = reconstruct_front(enc, flame, fip, flp)  # front geometry + landmarks
    cols, viss = [], []
    for i, vv in enumerate(a.views):
        ip, lp = vv.split(':')
        vcol, vis = bake_view(Vgeo, F, lmk51, ip, lp)
        cols.append(vcol); viss.append(vis)
        print(f"  [view {i}] {os.path.basename(ip)}: {(vis>0).sum()} verts visible")
    viss = np.stack(viss, 0); cols = np.stack(cols, 0)  # (Nviews, Nv), (Nviews, Nv, 3)
    best = np.argmax(viss, 0)  # per vertex, best view
    vcol = cols[best, np.arange(cols.shape[1])]
    seen = viss.max(0) > 0
    if (~seen).any() and seen.any():  # unseen by any view -> nearest seen + skin median
        skin = np.median(vcol[seen], 0)
        tree = cKDTree(Vgeo[seen]); dist, nn = tree.query(Vgeo[~seen])
        w = np.clip(dist / (0.5 * np.linalg.norm(Vgeo.max(0) - Vgeo.min(0))), 0, 1)[:, None]
        vcol[~seen] = (1 - w) * vcol[seen][nn] + w * skin
    print(f"  combined: {int(seen.sum())}/{len(Vgeo)} verts textured from real views")
    c = Vgeo.mean(0); Vc = Vgeo - c; scale = 0.9 * (a.size / 2) / np.abs(Vc[:, :2]).max()
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.size, a.size))
    for i in range(a.frames):
        th = 2 * np.pi * i / a.frames if a.max_deg >= 360 else np.deg2rad(a.max_deg) * np.sin(2 * np.pi * i / a.frames)
        Vr = Vc @ rot_y(th).T
        vx = Vr.copy(); vx[:, 0] = Vr[:, 0] * scale + a.size / 2; vx[:, 1] = -Vr[:, 1] * scale + a.size / 2; vx[:, 2] = Vr[:, 2]
        fidx, bary = raster(vx, F, a.size, a.size)
        canvas = np.full((a.size, a.size, 3), 0.2, np.float32)
        v = fidx >= 0
        if v.any():
            fi = fidx[v]; canvas[v] = np.einsum('mc,mcd->md', bary[v], vcol[F[fi]])
        fr = (canvas[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8); wr.write(fr)
        if i == 0:
            cv2.imwrite(os.path.splitext(a.out)[0] + "_f0.png", fr)
        if i == a.frames // 4:
            cv2.imwrite(os.path.splitext(a.out)[0] + "_q.png", fr)
    wr.release(); print("done ->", a.out)


if __name__ == '__main__':
    main()
