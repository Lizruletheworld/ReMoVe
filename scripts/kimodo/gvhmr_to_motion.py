#!/usr/bin/env python3
import os, sys, argparse
import numpy as np
import torch


from kimodo.skeleton import SMPLXSkeleton22, fk
from kimodo.motion_rep.feet import foot_detect_from_pos_and_vel
from kimodo.motion_rep.smooth_root import get_smooth_root_pos
from kimodo.constraints import compute_global_heading

FPS = 30.0
SMPL_TPOSE_PELVIS_Y = -0.3723  # kimodo.skeleton SMPLX22 neutral_joints[0][1]


def rotvec_to_mat(r):
    angle = np.linalg.norm(r)
    if angle < 1e-8:
        return np.eye(3)
    a = r / angle
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def mat_to_rotvec(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return ax * angle


def align_heading(go, transl):
    """Zero out the first-frame yaw and re-origin XZ, same convention as
    motionx_to_kimodo_gt.py / force_high_quality_select.py."""
    R0 = rotvec_to_mat(go[0])
    fwd = R0[:, 2]
    yaw = np.arctan2(fwd[0], fwd[2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ryaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rinv = Ryaw.T
    origin = transl[0].copy()
    origin[1] = 0.0
    go_out = np.empty_like(go)
    tr_out = np.empty_like(transl)
    for i in range(len(go)):
        go_out[i] = mat_to_rotvec(Rinv @ rotvec_to_mat(go[i]))
        tr_out[i] = Rinv @ (transl[i] - origin)
    return go_out, tr_out


def convert_one(hmr4d_path, skeleton, align=True):
    data = torch.load(hmr4d_path, map_location="cpu", weights_only=False)
    params = data["smpl_params_global"]
    body_pose = params["body_pose"].numpy()
    global_orient = params["global_orient"].numpy()
    transl = params["transl"].numpy()

    if global_orient.ndim == 3:
        global_orient = global_orient[:, 0, :]
    if body_pose.ndim == 2:
        body_pose = body_pose.reshape(body_pose.shape[0], 21, 3)

    if align:
        global_orient, transl = align_heading(global_orient, transl)

    T = global_orient.shape[0]
    aa = np.concatenate([global_orient[:, None, :], body_pose], axis=1)  # (T, 22, 3)
    local_rot_mats = torch.stack(
        [torch.tensor(np.stack([rotvec_to_mat(aa[t, j]) for j in range(22)]), dtype=torch.float32)
         for t in range(T)]
    )  # (T, 22, 3, 3)

    root_positions = torch.tensor(transl, dtype=torch.float32).clone()
    root_positions[:, 1] += SMPL_TPOSE_PELVIS_Y

    global_rot_mats, posed_joints, _ = fk(local_rot_mats, root_positions, skeleton,
                                           root_positions_is_global=True)

    smooth_root_pos = get_smooth_root_pos(root_positions.unsqueeze(0)).squeeze(0)

    pos_bt = posed_joints.unsqueeze(0)
    vel_bt = torch.zeros_like(pos_bt)
    if T > 1:
        vel_bt[:, :-1] = (pos_bt[:, 1:] - pos_bt[:, :-1]) * FPS
        vel_bt[:, -1] = vel_bt[:, -2]
    foot_contacts = foot_detect_from_pos_and_vel(
        pos_bt, vel_bt, skeleton, vel_thres=0.15, height_thresh=0.10
    ).squeeze(0).bool()

    global_root_heading = compute_global_heading(posed_joints.unsqueeze(0), skeleton).squeeze(0)

    return dict(
        local_rot_mats=local_rot_mats.numpy().astype(np.float32),
        global_rot_mats=global_rot_mats.numpy().astype(np.float32),
        posed_joints=posed_joints.numpy().astype(np.float32),
        root_positions=root_positions.numpy().astype(np.float32),
        smooth_root_pos=smooth_root_pos.numpy().astype(np.float32),
        foot_contacts=foot_contacts.numpy(),
        global_root_heading=global_root_heading.numpy().astype(np.float32),
    ), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+",
                     help="one or more <output_stem>=<hmr4d_results.pt path> pairs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    skeleton = SMPLXSkeleton22()

    n_ok = n_skip = n_fail = 0
    for item in args.items:
        if "=" not in item:
            print(f"[FAIL] bad item (expected name=path): {item}")
            n_fail += 1
            continue
        name, hmr4d_path = item.split("=", 1)
        out_path = os.path.join(args.out_dir, f"{name}.npz")
        if os.path.exists(out_path) and not args.force:
            print(f"[skip] {name} -> already present")
            n_skip += 1
            continue
        if not os.path.exists(hmr4d_path):
            print(f"[FAIL] {name}: MISSING hmr4d_results.pt: {hmr4d_path}")
            n_fail += 1
            continue
        arrays, T = convert_one(hmr4d_path, skeleton)
        np.savez(out_path, **arrays)
        print(f"[ok]   {name}: {T} frames -> {out_path}")
        n_ok += 1

    print(f"\nconverted={n_ok} skipped={n_skip} failed={n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
