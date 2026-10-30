#!/usr/bin/env python3
import argparse
import numpy as np
import cv2


def _axis_vec(axis):
    return {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}[axis]


def _ease(s, kind="linear"):
    if kind == "ease_in_out":
        return 0.5 - 0.5 * np.cos(np.pi * s)
    if kind == "ease_in":
        return s * s
    if kind == "ease_out":
        return 1 - (1 - s) * (1 - s)
    return s  # linear


def _orbit_pose(P, axis, theta_rad, radius_scale=1.0):
    """Documentation omitted in the public release."""
    axis_v = _axis_vec(axis).astype(np.float32)
    R = cv2.Rodrigues((axis_v * theta_rad).astype(np.float32))[0].astype(np.float32)
    t_move = (P - radius_scale * (R @ P)).astype(np.float32)
    return R, t_move


def traj_static(T, **kw):
    """Documentation omitted in the public release."""
    return [(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)) for _ in range(T)]


def traj_dolly(T, distance_m=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        t_move = np.array([0, 0, distance_m * s], dtype=np.float32)
        traj.append((np.eye(3, dtype=np.float32), t_move))
    return traj


def traj_truck(T, distance_m=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        t_move = np.array([distance_m * s, 0, 0], dtype=np.float32)
        traj.append((np.eye(3, dtype=np.float32), t_move))
    return traj


def traj_pedestal(T, distance_m=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        t_move = np.array([0, distance_m * s, 0], dtype=np.float32)
        traj.append((np.eye(3, dtype=np.float32), t_move))
    return traj


def traj_pan(T, angle_deg=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        R = cv2.Rodrigues((_axis_vec("y") * np.deg2rad(angle_deg * s)).astype(np.float32))[0].astype(np.float32)
        traj.append((R, np.zeros(3, dtype=np.float32)))
    return traj


def traj_tilt(T, angle_deg=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        R = cv2.Rodrigues((_axis_vec("x") * np.deg2rad(angle_deg * s)).astype(np.float32))[0].astype(np.float32)
        traj.append((R, np.zeros(3, dtype=np.float32)))
    return traj


def traj_orbit(T, P=None, axis="y", angle_deg=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    P = np.zeros(3, dtype=np.float32) if P is None else P
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        theta = np.deg2rad(angle_deg * s)
        traj.append(_orbit_pose(P, axis, theta, radius_scale=1.0))
    return traj


def traj_arc_dolly(T, P=None, axis="y", angle_deg=0.0, radius_frac=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    P = np.zeros(3, dtype=np.float32) if P is None else P
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        theta = np.deg2rad(angle_deg * s)
        r_scale = 1.0 - radius_frac * s
        traj.append(_orbit_pose(P, axis, theta, radius_scale=r_scale))
    return traj


def traj_arc_swoop(T, P=None, axis="y", angle_deg=0.0, dip_frac=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    P = np.zeros(3, dtype=np.float32) if P is None else P
    traj = []
    for i in range(T):
        s = i / max(T - 1, 1)
        s_eased = _ease(s, ease)
        theta = np.deg2rad(angle_deg * (s_eased - 0.5))
        r_scale = 1.0 - dip_frac * np.sin(np.pi * s)
        traj.append(_orbit_pose(P, axis, theta, radius_scale=r_scale))
    return traj


def traj_crane(T, up_m=0.0, dolly_m=0.0, ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    traj = []
    for i in range(T):
        s = _ease(i / max(T - 1, 1), ease)
        t_move = np.array([0, up_m * s, dolly_m * s], dtype=np.float32)
        traj.append((np.eye(3, dtype=np.float32), t_move))
    return traj


def traj_dolly_pan(T, distance_m=0.0, angle_deg=0.0, dolly_frac=0.6, pan_start_frame=None,
                    ease="ease_in_out", **kw):
    """Documentation omitted in the public release."""
    if pan_start_frame is not None:
        n_dolly = max(1, min(int(pan_start_frame), T - 1))
    else:
        n_dolly = max(1, int(round(T * dolly_frac)))
    traj = []
    for i in range(T):
        s_dolly = _ease(min(i, n_dolly - 1) / max(n_dolly - 1, 1), ease)
        t_move = np.array([0, 0, distance_m * s_dolly], dtype=np.float32)
        if i < n_dolly:
            s_pan = 0.0
        else:
            s_pan = _ease((i - n_dolly) / max(T - n_dolly - 1, 1), ease)
        R = cv2.Rodrigues((_axis_vec("y") * np.deg2rad(angle_deg * s_pan)).astype(np.float32))[0].astype(np.float32)
        traj.append((R, t_move))
    return traj


def traj_figure8(T, amplitude_x=0.0, amplitude_z=0.0, cycles=1.0, **kw):
    """Documentation omitted in the public release."""
    traj = []
    for i in range(T):
        s = i / max(T - 1, 1)
        phase = 2 * np.pi * cycles * s
        t_move = np.array([
            amplitude_x * np.sin(phase),
            0,
            amplitude_z * np.sin(2 * phase) * 0.5,
        ], dtype=np.float32)
        traj.append((np.eye(3, dtype=np.float32), t_move))
    return traj


def traj_handheld(T, pos_jitter_m=0.01, rot_jitter_deg=1.0, seed=0, **kw):
    """Documentation omitted in the public release."""
    rng = np.random.RandomState(seed)
    freqs = rng.uniform(0.5, 1.5, size=6)
    phases = rng.uniform(0, 2 * np.pi, size=6)
    traj = []
    for i in range(T):
        s = i / max(T - 1, 1)
        t_move = np.array([
            pos_jitter_m * np.sin(2 * np.pi * freqs[0] * s + phases[0]),
            pos_jitter_m * np.sin(2 * np.pi * freqs[1] * s + phases[1]),
            pos_jitter_m * 0.5 * np.sin(2 * np.pi * freqs[2] * s + phases[2]),
        ], dtype=np.float32)
        euler = np.array([
            np.deg2rad(rot_jitter_deg) * np.sin(2 * np.pi * freqs[3] * s + phases[3]),
            np.deg2rad(rot_jitter_deg) * np.sin(2 * np.pi * freqs[4] * s + phases[4]),
            np.deg2rad(rot_jitter_deg) * 0.3 * np.sin(2 * np.pi * freqs[5] * s + phases[5]),
        ], dtype=np.float32)
        R = cv2.Rodrigues(euler)[0].astype(np.float32)
        traj.append((R, t_move))
    return traj


TRAJECTORY_PRESETS = {
    "static":    traj_static,
    "dolly":     traj_dolly,
    "truck":     traj_truck,
    "pedestal":  traj_pedestal,
    "pan":       traj_pan,
    "tilt":      traj_tilt,
    "orbit":     traj_orbit,
    "arc_dolly": traj_arc_dolly,
    "arc_swoop": traj_arc_swoop,
    "dolly_pan": traj_dolly_pan,
    "crane":     traj_crane,
    "figure8":   traj_figure8,
    "handheld":  traj_handheld,
}


def compose_trajectories(traj_list):
    """Documentation omitted in the public release."""
    T = len(traj_list[0])
    out = []
    for i in range(T):
        R_total = np.eye(3, dtype=np.float32)
        t_total = np.zeros(3, dtype=np.float32)
        for traj in traj_list:
            R_i, t_i = traj[i]
            t_total = t_total + t_i
            R_total = R_i @ R_total
        out.append((R_total, t_total))
    return out


def apply_camera_move(points, R_move, t_move):
    """Documentation omitted in the public release."""
    return (R_move.T @ (points - t_move).T).T


#  mesh  

def face_shading(verts_cam, faces, skin_rgb, skin_shade, head_mask=None, head_colors=None):
    """Documentation omitted in the public release."""
    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    nz = np.abs(fn[:, 2]) / (np.linalg.norm(fn, axis=1) + 1e-9)
    shade = (1.0 - skin_shade) + skin_shade * nz
    colors = skin_rgb[None, :] * shade[:, None]
    if head_mask is not None:
        is_head_face = head_mask[faces].all(axis=1)
        head_tri_colors = head_colors[faces].mean(axis=1)
        colors[is_head_face] = head_tri_colors[is_head_face]
    return colors


def estimate_head_vertex_mask(verts0, head_frac=0.16):
    """Documentation omitted in the public release."""
    y = verts0[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    thresh = y_min + head_frac * (y_max - y_min)
    return y <= thresh


def bake_vertex_colors_from_frame(verts0, K, frame_bgr):
    """Documentation omitted in the public release."""
    H, W = frame_bgr.shape[:2]
    proj = (K @ verts0.T).T
    z = proj[:, 2]
    valid = z > 1e-3
    uv = np.zeros((verts0.shape[0], 2), dtype=np.float32)
    uv[valid] = proj[valid, :2] / z[valid, None]
    x, y = uv[:, 0], uv[:, 1]
    inside = valid & (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)

    x0 = np.floor(np.clip(x, 0, W - 2)).astype(np.int32)
    y0 = np.floor(np.clip(y, 0, H - 2)).astype(np.int32)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    frame_f = frame_bgr.astype(np.float32)
    c00 = frame_f[y0, x0]
    c10 = frame_f[y0, x0 + 1]
    c01 = frame_f[y0 + 1, x0]
    c11 = frame_f[y0 + 1, x0 + 1]
    bgr = c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy
    rgb01 = (bgr[:, ::-1] / 255.0).astype(np.float32)
    return rgb01, inside


def read_first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f": {path}")
    return frame


def render_mesh(verts_cam, faces, tri_colors, K, H, W):
    """Documentation omitted in the public release."""
    proj = (K @ verts_cam.T).T
    z = proj[:, 2]
    valid_v = z > 1e-3
    uv = np.zeros((verts_cam.shape[0], 2), dtype=np.float32)
    uv[valid_v] = proj[valid_v, :2] / z[valid_v, None]

    tri_z = z[faces].mean(axis=1)             # (F,) 
    tri_valid = valid_v[faces].all(axis=1)    # 
    tri_uv = uv[faces]                        # (F,3,2)

    order = np.argsort(tri_z)[::-1]           # (z)->(z)
    face_order = order[tri_valid[order]]

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    for fi in face_order:
        pts = np.round(tri_uv[fi]).astype(np.int32).reshape(-1, 1, 2)
        r, g, b = (tri_colors[fi] * 255).clip(0, 255)
        cv2.fillConvexPoly(canvas, pts, (int(b), int(g), int(r)))  # RGB->BGR
        cv2.fillConvexPoly(mask, pts, 1)

    return canvas, mask.astype(bool)


def render_mesh_textured(verts_cam, faces, tri_colors, K, H, W,
                          head_mask=None, head_valid=None, verts0=None, ref_frame0=None,
                          depth0_zbuf=None):
    """Documentation omitted in the public release."""
    proj = (K @ verts_cam.T).T
    z = proj[:, 2]
    valid_v = z > 1e-3
    uv = np.zeros((verts_cam.shape[0], 2), dtype=np.float32)
    uv[valid_v] = proj[valid_v, :2] / z[valid_v, None]

    tri_z = z[faces].mean(axis=1)
    tri_valid = valid_v[faces].all(axis=1)
    tri_uv = uv[faces]  # (F,3,2) 

    use_texture = np.zeros(faces.shape[0], dtype=bool)
    tri_uv0 = None
    if head_mask is not None and verts0 is not None and ref_frame0 is not None:
        is_head_face = head_mask[faces].all(axis=1)
        proj0 = (K @ verts0.T).T
        z0 = proj0[:, 2]
        valid0 = z0 > 1e-3
        uv0 = np.zeros((verts0.shape[0], 2), dtype=np.float32)
        uv0[valid0] = proj0[valid0, :2] / z0[valid0, None]
        tri_uv0 = uv0[faces]  # (F,3,2) 0
        face_ref_ok = valid0[faces].all(axis=1)
        if head_valid is not None:
            face_ref_ok = face_ref_ok & head_valid[faces].all(axis=1)

        visible0 = np.ones(faces.shape[0], dtype=bool)
        if depth0_zbuf is not None:
            tri_z0 = z0[faces].mean(axis=1)
            cx = np.clip(np.round(tri_uv0[:, :, 0].mean(axis=1)), 0, W - 1).astype(np.int32)
            cy = np.clip(np.round(tri_uv0[:, :, 1].mean(axis=1)), 0, H - 1).astype(np.int32)
            zbuf_at_face = depth0_zbuf[cy, cx]
            visible0 = tri_z0 <= zbuf_at_face * 1.02 + 1e-6  # 2%

        use_texture = is_head_face & face_ref_ok & visible0

    order = np.argsort(tri_z)[::-1]
    face_order = order[tri_valid[order]]

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    ref_h, ref_w = (ref_frame0.shape[:2] if ref_frame0 is not None else (0, 0))

    for fi in face_order:
        dst_tri = tri_uv[fi].astype(np.float32)          # (3,2) 
        pts_i = np.round(dst_tri).astype(np.int32).reshape(-1, 1, 2)

        warped_ok = False
        if use_texture[fi]:
            src_tri = tri_uv0[fi].astype(np.float32)
            area2 = abs((dst_tri[1, 0] - dst_tri[0, 0]) * (dst_tri[2, 1] - dst_tri[0, 1])
                        - (dst_tri[2, 0] - dst_tri[0, 0]) * (dst_tri[1, 1] - dst_tri[0, 1]))
            src_ok = (src_tri[:, 0].min() >= 0 and src_tri[:, 0].max() <= ref_w - 1
                      and src_tri[:, 1].min() >= 0 and src_tri[:, 1].max() <= ref_h - 1)
            if area2 >= 1.0 and src_ok:
                x, y, w, h = cv2.boundingRect(pts_i)
                x, y = max(x, 0), max(y, 0)
                w, h = min(w, W - x), min(h, H - y)
                if w > 0 and h > 0:
                    M = cv2.getAffineTransform(src_tri, dst_tri - np.array([x, y], dtype=np.float32))
                    warped = cv2.warpAffine(ref_frame0, M, (w, h),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    tri_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(tri_mask, (pts_i - np.array([[x, y]])).astype(np.int32), 1)
                    tb = tri_mask.astype(bool)
                    canvas[y:y + h, x:x + w][tb] = warped[tb]
                    mask[y:y + h, x:x + w][tb] = 1
                    warped_ok = True

        if not warped_ok:
            r, g, b = (tri_colors[fi] * 255).clip(0, 255)
            cv2.fillConvexPoly(canvas, pts_i, (int(b), int(g), int(r)))
            cv2.fillConvexPoly(mask, pts_i, 1)

    return canvas, mask.astype(bool)


def render_mesh_depth(verts_cam, faces, K, H, W):
    """Documentation omitted in the public release."""
    proj = (K @ verts_cam.T).T
    z = proj[:, 2]
    valid_v = z > 1e-3
    uv = np.zeros((verts_cam.shape[0], 2), dtype=np.float32)
    uv[valid_v] = proj[valid_v, :2] / z[valid_v, None]

    tri_z = z[faces].mean(axis=1)
    tri_valid = valid_v[faces].all(axis=1)
    tri_uv = uv[faces]

    order = np.argsort(tri_z)[::-1]
    face_order = order[tri_valid[order]]

    canvas = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for fi in face_order:
        pts = np.round(tri_uv[fi]).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(canvas, pts, float(tri_z[fi]))
        cv2.fillConvexPoly(mask, pts, 1)
    return canvas, mask.astype(bool)


def unproject_depth(depth, K):
    """Documentation omitted in the public release."""
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    K_inv = np.linalg.inv(K)
    uv1 = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32).reshape(-1, 3)
    rays = (K_inv @ uv1.T).T
    points = rays * depth.reshape(-1, 1)
    return points.reshape(H, W, 3)


def splat_render_scalar(points_cam, values, K, H, W, bg_value=0.0,
                         splat_offsets=((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
    """Documentation omitted in the public release."""
    proj = (K @ points_cam.T).T
    z = proj[:, 2]
    valid = z > 1e-3
    uv = proj[valid, :2] / z[valid, None]
    z_v = z[valid]
    c_v = values[valid]

    u_px = np.round(uv[:, 0]).astype(np.int32)
    v_px = np.round(uv[:, 1]).astype(np.int32)
    ib = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
    u_px, v_px, z_v, c_v = u_px[ib], v_px[ib], z_v[ib], c_v[ib]

    order = np.argsort(z_v)[::-1]
    u_px, v_px, c_v = u_px[order], v_px[order], c_v[order]

    canvas = np.full(H * W, bg_value, dtype=np.float32)
    mask = np.zeros(H * W, dtype=bool)
    for dy, dx in splat_offsets:
        uu = np.clip(u_px + dx, 0, W - 1)
        vv = np.clip(v_px + dy, 0, H - 1)
        idx = vv * W + uu
        canvas[idx] = c_v
        mask[idx] = True
    return canvas.reshape(H, W), mask.reshape(H, W)


def inpaint_scalar_holes(canvas, mask, inpaint_radius=7):
    """Documentation omitted in the public release."""
    if mask.all():
        return canvas
    valid_vals = canvas[mask]
    if valid_vals.size == 0:
        return canvas
    d_min, d_max = valid_vals.min(), valid_vals.max()
    holes = (~mask).astype(np.uint8) * 255
    d_u8 = (((canvas - d_min) / (d_max - d_min + 1e-8)).clip(0, 1) * 255).astype(np.uint8)
    d_u8 = cv2.inpaint(d_u8, holes, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    return (d_u8.astype(np.float32) / 255.0) * (d_max - d_min) + d_min


def calibrate_mesh_depth_scale(verts0, faces, K, H, W, bg_depth0, ring_px=25):
    """Documentation omitted in the public release."""
    mesh_depth0, mesh_mask0 = render_mesh_depth(verts0, faces, K, H, W)
    if not mesh_mask0.any():
        print("[calib][WARN] mesh0scale=1.0")
        return 1.0
    mesh_centroid_depth = float(np.median(mesh_depth0[mesh_mask0]))

    ys, xs = np.where(mesh_mask0)
    y_bottom, x_lo, x_hi = int(ys.max()), int(xs.min()), int(xs.max())
    strip = bg_depth0[y_bottom + 1:min(y_bottom + 1 + ring_px, H), x_lo:x_hi + 1]
    if strip.size < 50:
        print("[calib][WARN] scale=1.0")
        return 1.0
    target_depth = float(np.median(strip))
    scale = target_depth / (mesh_centroid_depth + 1e-8)
    print(f"[calib] mesh={mesh_centroid_depth:.3f}m  ={target_depth:.3f}m  "
          f"-> scale={scale:.4f}")
    if scale <= 0 or not np.isfinite(scale):
        print("[calib][WARN] scale1.0")
        return 1.0
    return scale


def estimate_skin_rgb_from_video(path, sample_frames=5, bg_diff_thresh=20):
    """Documentation omitted in the public release."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raise FileNotFoundError(f": {path}")
    idxs = np.linspace(0, n - 1, min(sample_frames, n)).astype(int)
    bright_bgr = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        bg_color = frame[0, 0].astype(np.int16)
        diff = np.abs(frame.astype(np.int16) - bg_color).sum(axis=2)
        person = diff > bg_diff_thresh
        if person.sum() < 200:
            continue
        pixels = frame[person].astype(np.float32)
        brightness = pixels.sum(axis=1)
        thresh = np.percentile(brightness, 95)
        bright_bgr.append(pixels[brightness >= thresh].mean(axis=0))
    cap.release()
    if not bright_bgr:
        raise RuntimeError(f" {path}  --skin_rgb ")
    bgr = np.mean(bright_bgr, axis=0)
    rgb01 = (bgr[::-1] / 255.0).astype(np.float32)
    print(f"[info]  {path}  skin_rgb(RGB 0-1) = {rgb01.tolist()}")
    return rgb01


#   

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--front_back_cache", required=True,
                     help="mesh-conditioning renderer  "
                          "*_front_back_cache.npz (verts_cam_all/faces/K/W/H)")
    ap.add_argument("--reference_video", required=True,
                     help=" (mesh_v9_skin/{NAME}.mp4)(a) "
                          "--skin_rgb (b)0")
    ap.add_argument("--skin_rgb", default=None,
                     help=" --skin_body  'R,G,B' (0-255) "
                          "--reference_video ")
    ap.add_argument("--skin_shade", type=float, default=0.45,
                     help=" SHADE  (v9_skinbody_batch_offline5.sh 0.45)")
    ap.add_argument("--head_frac", type=float, default=0.16,
                     help="mesh()")
    ap.add_argument("--person_bg_color", default="128,128,128",
                     help=" 'R,G,B' (0-255)")
    ap.add_argument("--bg_depth_npz", default=None,
                     help="[] bake_clean_bg_depth.py npz(depth (H,W))"
                          " fused_depth_moved_turbo.mp4DIBR"
                          "mesh(DA3)mask"
                          "()")
    ap.add_argument("--depth_inpaint_radius", type=int, default=7, help="disocclusioncv2.inpaint")
    ap.add_argument("--calib_ring_px", type=int, default=25,
                     help="[ --bg_depth_npz ] mesh()")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=int, default=24)

    ap.add_argument("--preset", default="dolly", choices=list(TRAJECTORY_PRESETS.keys()),
                     help="")
    ap.add_argument("--ease", default="ease_in_out", choices=["linear", "ease_in", "ease_out", "ease_in_out"])

    # dolly / truck / pedestal 
    ap.add_argument("--distance_m", type=float, default=0.0, help="()")
    # pan / tilt / orbit / arc_dolly 
    ap.add_argument("--angle_deg", type=float, default=0.0, help="()")
    ap.add_argument("--axis", default="y", choices=["x", "y", "z"], help="orbit/arc_dolly ")
    # arc_dolly 
    ap.add_argument("--radius_frac", type=float, default=0.0, help="/")
    # arc_swoop angle_deg(-angle_deg/2+angle_deg/2)
    ap.add_argument("--dip_frac", type=float, default=0.0,
                     help="arc_swoop: (0.2=20%)"
                          "")
    # dolly_pan distance_m/angle_deg 
    ap.add_argument("--dolly_frac", type=float, default=0.6,
                     help="dolly_pan: ")
    ap.add_argument("--pan_start_frame", type=int, default=None,
                     help="dolly_pan: --dolly_frac"
                          "()--num_frames")
    ap.add_argument("--num_frames", type=int, default=None,
                     help="meshN()9797")
    # orbit / arc_dolly mesh
    ap.add_argument("--look_at_z", type=float, default=None, help="()")
    # crane 
    ap.add_argument("--crane_up_m", type=float, default=0.0, help="crane: ()=")
    ap.add_argument("--crane_forward_m", type=float, default=0.0, help="crane: ()")
    # figure8 
    ap.add_argument("--amp_x", type=float, default=0.0, help="figure8: x()")
    ap.add_argument("--amp_z", type=float, default=0.0, help="figure8: z()")
    ap.add_argument("--cycles", type=float, default=1.0, help="figure8: ")
    # 
    ap.add_argument("--add_handheld", action="store_true", help="")
    ap.add_argument("--handheld_pos_m", type=float, default=0.01)
    ap.add_argument("--handheld_rot_deg", type=float, default=1.0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    person_bg_bgr = tuple(int(c) for c in reversed(args.person_bg_color.split(",")))  # RGB(str)->BGR(int)

    if args.skin_rgb is not None:
        skin_rgb = np.array([float(x) for x in args.skin_rgb.split(",")], dtype=np.float32) / 255.0
    else:
        skin_rgb = estimate_skin_rgb_from_video(args.reference_video)

    cache = np.load(args.front_back_cache)
    verts_cam_all = cache["verts_cam_all"].astype(np.float32)   # (T, V, 3)
    faces = cache["faces"].astype(np.int64)                     # (F, 3)
    K = cache["K"].astype(np.float32)                           # (3, 3)
    W, H = int(cache["W"]), int(cache["H"])
    if args.num_frames is not None and args.num_frames > 0:
        verts_cam_all = verts_cam_all[:args.num_frames]
    T = verts_cam_all.shape[0]
    print(f"[info] T={T} frames, H={H} W={W}")

    # () + meshP/
    # verts_cam_all(02D
    # )PorbitP
    # mesh
    bg_depth_static = None
    if args.bg_depth_npz is not None:
        with np.load(args.bg_depth_npz) as d:
            bg_depth_static = d["depth"].astype(np.float32)
        if bg_depth_static.shape != (H, W):
            bg_depth_static = cv2.resize(bg_depth_static, (W, H), interpolation=cv2.INTER_LINEAR)
        depth_scale = calibrate_mesh_depth_scale(
            verts_cam_all[0], faces, K, H, W, bg_depth_static, ring_px=args.calib_ring_px)
        verts_cam_all = verts_cam_all * depth_scale

    # 0
    ref_frame0 = read_first_frame(args.reference_video)
    head_mask = estimate_head_vertex_mask(verts_cam_all[0], args.head_frac)
    head_colors, head_valid = bake_vertex_colors_from_frame(verts_cam_all[0], K, ref_frame0)
    head_colors[~head_valid] = skin_rgb
    print(f"[info] : {int(head_mask.sum())}/{len(head_mask)}"
          f" {int((head_mask & ~head_valid).sum())} 0skin_rgb")
    # 0meshz-bufferrender_mesh_textured0
    # (/)(/)
    depth0_zbuf, _ = render_mesh_depth(verts_cam_all[0], faces, K, H, W)

    #  P0meshorbit / arc_dolly 
    if args.look_at_z is not None:
        P = np.array([0, 0, args.look_at_z], dtype=np.float32)
    else:
        P = verts_cam_all[0].mean(axis=0).astype(np.float32)
    print(f"[info]  P () = {P}")

    preset_fn = TRAJECTORY_PRESETS[args.preset]
    traj = preset_fn(
        T, P=P, axis=args.axis, angle_deg=args.angle_deg, distance_m=args.distance_m,
        radius_frac=args.radius_frac, up_m=args.crane_up_m, dolly_m=args.crane_forward_m,
        amplitude_x=args.amp_x, amplitude_z=args.amp_z, cycles=args.cycles, ease=args.ease,
        dolly_frac=args.dolly_frac, pan_start_frame=args.pan_start_frame, dip_frac=args.dip_frac,
    )
    if args.add_handheld:
        hh = traj_handheld(T, pos_jitter_m=args.handheld_pos_m, rot_jitter_deg=args.handheld_rot_deg)
        traj = compose_trajectories([traj, hh])

    depth_writer = None
    if bg_depth_static is not None:
        depth_writer = cv2.VideoWriter(os.path.join(args.out_dir, "fused_depth_moved_turbo.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb_writer = cv2.VideoWriter(os.path.join(args.out_dir, "person_rgb_moved.mp4"), fourcc, args.fps, (W, H))
    mask_writer = cv2.VideoWriter(os.path.join(args.out_dir, "person_mask_moved.mp4"), fourcc, args.fps,
                                   (W, H), isColor=False)

    for t in range(T):
        R_move, t_move = traj[t]
        verts_new = apply_camera_move(verts_cam_all[t], R_move, t_move)
        tri_colors = face_shading(verts_new, faces, skin_rgb, args.skin_shade,
                                   head_mask=head_mask, head_colors=head_colors)
        person_rgb, person_mask = render_mesh_textured(
            verts_new, faces, tri_colors, K, H, W,
            head_mask=head_mask, head_valid=head_valid,
            verts0=verts_cam_all[0], ref_frame0=ref_frame0, depth0_zbuf=depth0_zbuf)

        final_rgb = np.full((H, W, 3), person_bg_bgr, dtype=np.uint8)
        final_rgb[person_mask] = person_rgb[person_mask]

        rgb_writer.write(final_rgb)
        mask_writer.write(person_mask.astype(np.uint8) * 255)

        if bg_depth_static is not None:
            # mesh(RGBDA3)
            person_depth, person_depth_mask = render_mesh_depth(verts_new, faces, K, H, W)
            # DIBR(meshsplat)
            pts_bg_old = unproject_depth(bg_depth_static, K).reshape(-1, 3)
            pts_bg_new = apply_camera_move(pts_bg_old, R_move, t_move)
            bg_depth_new_raw, bg_covered = splat_render_scalar(
                pts_bg_new, bg_depth_static.reshape(-1), K, H, W, bg_value=0.0)
            bg_depth_new = inpaint_scalar_holes(bg_depth_new_raw, bg_covered, args.depth_inpaint_radius)

            final_depth = bg_depth_new.copy()
            final_depth[person_depth_mask] = person_depth[person_depth_mask]

            d_min, d_max = final_depth.min(), final_depth.max()
            norm = (((final_depth - d_min) / (d_max - d_min + 1e-8)).clip(0, 1) * 255).astype(np.uint8)
            depth_writer.write(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO))

        print(f"[{t + 1}/{T}] done", flush=True)

    rgb_writer.release()
    mask_writer.release()
    if depth_writer is not None:
        depth_writer.release()
    print(f"\n[DONE] : {args.out_dir}")


if __name__ == "__main__":
    main()
