import os, csv, argparse
import numpy as np
import torch

from paths import SMPLX_PATH
DEFAULT_SMPLX = SMPLX_PATH
FPS           = 30.0
CONTACT_H     = 0.05
GROUND_PCT    = 2       # v2: 2nd percentile (was 5th) -- better for jumping motions
N_JOINTS      = 22


def load_smplx(path):
    m = np.load(path, allow_pickle=True)
    parents = m["kintree_table"][0, :N_JOINTS].astype(np.int32)
    parents[0] = -1
    return parents, m["v_template"].astype(np.float64), m["J_regressor"][:N_JOINTS], m["shapedirs"]


def shaped_joints(v_tpl, J_reg, sdirs, betas):
    b = betas.mean(0) if betas.ndim == 2 else betas
    v = v_tpl + np.einsum("vdi,i->vd", sdirs[:, :, :10], b.astype(np.float64))
    return (J_reg @ v).astype(np.float32)


def aa2rotmat(aa):
    T     = aa.shape[0]
    angle = np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-12
    axis  = (aa / angle).astype(np.float32)
    angle = angle[:, 0].astype(np.float32)
    c = np.cos(angle)[:, None, None]
    s = np.sin(angle)[:, None, None]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    K = np.zeros((T, 3, 3), np.float32)
    K[:,0,1]=-z; K[:,0,2]=y; K[:,1,0]=z; K[:,1,2]=-x; K[:,2,0]=-y; K[:,2,1]=x
    I  = np.eye(3, dtype=np.float32)[None]
    nn = axis[:,:,None] * axis[:,None,:]
    return c*I + (1-c)*nn + s*K


def smplx_fk(go, bp, tr, J_tpl, parents):
    T = go.shape[0]; N = len(parents)
    all_aa = np.concatenate([go[:,None,:], bp.reshape(T,-1,3)], axis=1)
    all_R  = np.stack([aa2rotmat(all_aa[:,i,:]) for i in range(N)], axis=1)
    pos = np.zeros((T, N, 3), np.float32)
    Rg  = np.zeros((T, N, 3, 3), np.float32)
    for i in range(N):
        p = int(parents[i])
        if p < 0:
            pos[:,i,:] = J_tpl[i]; Rg[:,i] = all_R[:,i]
        else:
            off = (J_tpl[i] - J_tpl[p]).astype(np.float32)
            pos[:,i,:] = pos[:,p,:] + np.einsum("tij,j->ti", Rg[:,p], off)
            Rg[:,i]    = np.einsum("tij,tjk->tik", Rg[:,p], all_R[:,i])
    pos += tr[:,None,:].astype(np.float32)
    return pos


def compute_metrics(joints, transl, fps=FPS, contact_h=CONTACT_H, gnd_pct=GROUND_PCT):
    dt    = 1.0 / fps
    l_h   = np.minimum(joints[:,7,1], joints[:,10,1])
    r_h   = np.minimum(joints[:,8,1], joints[:,11,1])
    min_h = np.minimum(l_h, r_h)
    ground_y = float(np.percentile(min_h, gnd_pct))
    l_con = l_h < ground_y + contact_h
    r_con = r_h < ground_y + contact_h
    any_con = l_con | r_con
    gp    = np.maximum(0.0, ground_y - min_h).astype(np.float32)
    fl    = np.where(any_con, np.maximum(0.0, min_h - ground_y), 0.0).astype(np.float32)
    l_spd = np.linalg.norm(np.diff(joints[:,7,[0,2]], axis=0), axis=-1) / dt
    r_spd = np.linalg.norm(np.diff(joints[:,8,[0,2]], axis=0), axis=-1) / dt
    l_fs  = np.where(l_con[:-1] & l_con[1:], l_spd, 0.0)
    r_fs  = np.where(r_con[:-1] & r_con[1:], r_spd, 0.0)
    fs    = np.concatenate([[0.0], np.maximum(l_fs, r_fs)]).astype(np.float32)
    vel      = np.diff(transl, axis=0) * fps
    vel_mag  = np.linalg.norm(vel, axis=-1)                              # [T-1]
    vel_out  = np.concatenate([[0.], vel_mag]).astype(np.float32)       # [T]
    acc      = np.diff(vel, axis=0) * fps
    accel    = np.concatenate([[0.,0.], np.linalg.norm(acc, axis=-1)]).astype(np.float32)
    jerk_vec = np.diff(acc, axis=0) * fps
    jerk     = np.concatenate([[0.,0.,0.], np.linalg.norm(jerk_vec, axis=-1)]).astype(np.float32)
    return dict(gp_m=gp, fl_m=fl, fs_mps=fs,
                root_vel_mps=vel_out,
                root_accel_mps2=accel, root_jerk_mps3=jerk,
                l_contact=l_con.astype(np.float32), r_contact=r_con.astype(np.float32),
                ground_y=ground_y,
                l_foot_h_m=l_h.astype(np.float32), r_foot_h_m=r_h.astype(np.float32))


def process_one(pt_path, parents, v_tpl, J_reg, sdirs):
    d     = torch.load(pt_path, map_location="cpu", weights_only=False)
    g     = d["smpl_params_global"]
    J_tpl = shaped_joints(v_tpl, J_reg, sdirs, g["betas"].numpy())
    joints = smplx_fk(g["global_orient"].numpy(), g["body_pose"].numpy(),
                      g["transl"].numpy(), J_tpl, parents)
    return compute_metrics(joints, g["transl"].numpy())


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root",   required=True, help="dir with <sample>/hmr4d_results.pt (GVHMR --output_root)")
    pa.add_argument("--out_dir",     required=True, help="where <sample>_per_frame.csv and summary.csv are written")
    pa.add_argument("--smplx_model", default=DEFAULT_SMPLX)
    args = pa.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    parents, v_tpl, J_reg, sdirs = load_smplx(args.smplx_model)
    seqs = sorted([d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)) and
                   os.path.exists(os.path.join(args.data_root, d, "hmr4d_results.pt"))])
    MKEYS = ["gp_m","fl_m","fs_mps","root_vel_mps","root_accel_mps2","root_jerk_mps3"]
    rows  = []
    print(f"{'Sequence':<35}  {'GP':>6}  {'FL':>6}  {'FS':>7}  {'Accel':>9}  {'Jerk':>8}  {'Con%':>5}  {'FS_con(med)':>11}")
    print("-"*100)
    for seq in seqs:
        pt = os.path.join(args.data_root, seq, "hmr4d_results.pt")
        m  = process_one(pt, parents, v_tpl, J_reg, sdirs)
        T  = len(m["gp_m"])
        with open(os.path.join(args.out_dir, f"{seq}_per_frame.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame","time_s","gp_m","fl_m","fs_mps",
                        "root_vel_mps","root_accel_mps2","root_jerk_mps3",
                        "l_contact","r_contact","l_foot_h_m","r_foot_h_m"])
            for t in range(T):
                w.writerow([t, round(t/FPS,4),
                            round(float(m["gp_m"][t]),6), round(float(m["fl_m"][t]),6),
                            round(float(m["fs_mps"][t]),6),
                            round(float(m["root_vel_mps"][t]),6),
                            round(float(m["root_accel_mps2"][t]),6),
                            round(float(m["root_jerk_mps3"][t]),6),
                            int(m["l_contact"][t]), int(m["r_contact"][t]),
                            round(float(m["l_foot_h_m"][t]),6), round(float(m["r_foot_h_m"][t]),6)])
        con_mask = (m["l_contact"]>0)|(m["r_contact"]>0)
        # v2 key fix: MEDIAN of contact-frame FS -- robust to the noisy outlier landing frames
        fs_con = float(np.median(m["fs_mps"][con_mask])) if con_mask.any() else 0.0
        row = {"sequence": seq, "n_frames": T, "ground_y_m": round(m["ground_y"],4)}
        for k in MKEYS:
            row[f"{k}_mean"] = round(float(m[k].mean()),6)
            row[f"{k}_max"]  = round(float(m[k].max()),6)
        row["contact_ratio"]       = round(float(con_mask.mean()),4)
        row["fs_contact_mean_mps"] = round(fs_con, 6)
        rows.append(row)
        print(f"{seq:<35}  {row['gp_m_mean']:>6.4f}  {row['fl_m_mean']:>6.4f}  "
              f"{row['fs_mps_mean']:>7.4f}  {row['root_accel_mps2_mean']:>9.3f}  "
              f"{row['root_jerk_mps3_mean']:>8.3f}  {row['contact_ratio']*100:>4.1f}%  "
              f"{fs_con:>11.4f}")
    keys = list(rows[0].keys())
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
