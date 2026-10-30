import argparse
import json
import os

import numpy as np
import torch

# same as gvhmr_quality_select_v2.py
SMPL_TPOSE_PELVIS_Y = -0.3723


def rotvec_to_matrix(r):
    angle = np.linalg.norm(r)
    if angle < 1e-8:
        return np.eye(3)
    axis = r / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def matrix_to_rotvec(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def extract_yaw(R):
    fwd = R[:, 2]
    yaw = np.arctan2(fwd[0], fwd[2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])


def align_heading(global_orient, transl):
    R0 = rotvec_to_matrix(global_orient[0])
    R_inv = extract_yaw(R0).T
    origin = transl[0].copy()
    origin[1] = 0.0
    aligned_go = np.zeros_like(global_orient)
    aligned_tr = np.zeros_like(transl)
    for i in range(len(global_orient)):
        aligned_go[i] = matrix_to_rotvec(R_inv @ rotvec_to_matrix(global_orient[i]))
        aligned_tr[i] = R_inv @ (transl[i] - origin)
    return aligned_go, aligned_tr


def process_sample_force_high(pt_path, align=True):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
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

    n_frames = global_orient.shape[0]
    frame_indices = list(range(n_frames))  # HIGH tier == all frames

    entry = {"type": "fullbody", "frame_indices": frame_indices,
             "local_joints_rot": [], "root_positions": [], "smooth_root_2d": []}
    for fi in frame_indices:
        full_pose = np.concatenate(
            [global_orient[fi].reshape(1, 3), body_pose[fi].reshape(21, 3)], axis=0)
        entry["local_joints_rot"].append(full_pose.astype(float).tolist())
        root_pos = transl[fi].copy()
        root_pos[1] += SMPL_TPOSE_PELVIS_Y
        entry["root_positions"].append(root_pos.astype(float).tolist())
        entry["smooth_root_2d"].append([float(root_pos[0]), float(root_pos[2])])

    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvhmr_dir", nargs="+", required=True,
                    help="one or more GVHMR output dirs (each containing hmr4d_results.pt)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_align_heading", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tiers = {}
    for d in args.gvhmr_dir:
        sample = os.path.basename(os.path.normpath(d))
        pt_path = os.path.join(d, "hmr4d_results.pt")
        if not os.path.exists(pt_path):
            print(f"  SKIP {sample}: missing {pt_path}")
            continue
        entry = process_sample_force_high(pt_path, align=not args.no_align_heading)
        out_path = os.path.join(args.out_dir, f"{sample}.json")
        with open(out_path, "w") as f:
            json.dump([entry], f, indent=2)
        tiers[sample] = "high"
        print(f"  {sample}: tier=high (forced)  -> {len(entry['frame_indices'])} frames  -> {out_path}")
    with open(os.path.join(args.out_dir, "tiers.json"), "w") as f:
        json.dump(tiers, f, indent=2)


if __name__ == "__main__":
    main()
