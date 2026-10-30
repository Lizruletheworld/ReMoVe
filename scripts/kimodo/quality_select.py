import argparse, csv, json, os
import numpy as np
import torch

GVHMR_ROOT  = None   # set from --gvhmr_root
CSV_ROOT    = None   # set from --csv_root
SUMMARY_CSV = None   # set from --summary_csv
OUT_DIR     = None


FS_CONTACT_HIGH  = 0.4
FS_CONTACT_LOW   = 1.0
FS_MAX_LOW       = 3.0
JERK_LOW         = 3000.0  # jerk_max above this -> LOW (too dynamic)
ACCEL_LOW        = 70.0   # accel_max above this -> LOW (too dynamic)
JERK_MAX_MEDIUM  = 2000.0
ACCEL_MAX_MEDIUM = 35.0

GP_T   = 0.02
FL_T   = 0.05
FS_T   = 1.5
JERK_T  = 2000.0
ACCEL_T = 15.0    # m/s  per-frame root accel threshold (MEDIUM)
VEL_T   = 5.0    # m/s   per-frame root velocity threshold (MEDIUM)
VEL_LOW = 8.0    # m/s   vel_max above this -> LOW tier

PREFIX_FRAMES = 5

# SMPL T-pose pelvis Y offset correction (same as Kimodo SMPLX22 neutral_joints[0][1])
SMPL_TPOSE_PELVIS_Y = -0.3723


#  Rotation helpers 
def rotvec_to_matrix(r):
    angle = np.linalg.norm(r)
    if angle < 1e-8:
        return np.eye(3)
    axis = r / angle
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*K@K

def matrix_to_rotvec(R):
    angle = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(angle))
    return axis * angle

def extract_yaw(R):
    fwd = R[:,2]
    yaw = np.arctan2(fwd[0], fwd[2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])

def align_heading(global_orient, transl):
    R0    = rotvec_to_matrix(global_orient[0])
    R_inv = extract_yaw(R0).T
    origin = transl[0].copy(); origin[1] = 0.0
    aligned_go = np.zeros_like(global_orient)
    aligned_tr = np.zeros_like(transl)
    for i in range(len(global_orient)):
        aligned_go[i] = matrix_to_rotvec(R_inv @ rotvec_to_matrix(global_orient[i]))
        aligned_tr[i] = R_inv @ (transl[i] - origin)
    return aligned_go, aligned_tr


#  CSV helpers 
def load_summary(path):
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            result[row["sequence"]] = row
    return result

def read_per_frame_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    result = {}
    for key in rows[0]:
        try:    result[key] = np.array([float(r[key]) for r in rows])
        except: result[key] = np.array([r[key] for r in rows])
    return result


#  Quality classification 
def sample_tier(row, fs_c_high, fs_c_low, fs_max_low, jerk_max_med, accel_max_med, jerk_low, accel_low, vel_low):
    fs_c  = float(row["fs_contact_mean_mps"])
    fs_mx = float(row["fs_mps_max"])
    jerk  = float(row["root_jerk_mps3_max"])
    accel = float(row["root_accel_mps2_max"])
    vel_mx = float(row["root_vel_mps_max"])
    if fs_c >= fs_c_low or fs_mx >= fs_max_low or jerk >= jerk_low or accel >= accel_low or vel_mx >= vel_low:
        return "low"
    if fs_c < fs_c_high and jerk < jerk_max_med and accel < accel_max_med:
        return "high"
    return "medium"

def medium_frame_mask(csv_data, gp_t, fl_t, fs_t, jerk_t, accel_t, vel_t):
    bad = ((csv_data["gp_m"] > gp_t) | (csv_data["fl_m"] > fl_t) |
           (csv_data["fs_mps"] > fs_t) | (csv_data["root_jerk_mps3"] > jerk_t) |
           (csv_data["root_accel_mps2"] > accel_t) | (csv_data["root_vel_mps"] > vel_t))
    return ~bad


#  Segment helpers (v2 new) 
def find_good_segments(good_mask, min_len=3, max_gap=2):
    """
    Convert boolean good-frame mask to list of (start, end) inclusive segments.
    Gaps of up to max_gap consecutive bad frames are bridged.
    Segments shorter than min_len are discarded.
    """
    n = len(good_mask)
    # Bridge small gaps
    filled = good_mask.copy()
    i = 0
    while i < n:
        if not filled[i]:
            j = i
            while j < n and not good_mask[j]:
                j += 1
            if (j - i) <= max_gap and i > 0 and j < n:
                filled[i:j] = True
            i = j
        else:
            i += 1
    # Collect runs
    segments = []
    start = None
    for i in range(n):
        if filled[i] and start is None:
            start = i
        elif not filled[i] and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, n - 1))
    return [(s, e) for s, e in segments if e - s + 1 >= min_len]


def select_frame_indices(tier, n_frames, per_frame_good, prefix_frames,
                         seg_min_len=3, seg_max_gap=2):
    if tier == "high":
        return list(range(n_frames))

    if tier == "low":
        indices = list(range(min(prefix_frames, n_frames)))
        # always include frame 0
        if 0 not in indices:
            indices = [0] + indices
        return sorted(set(indices))

    # MEDIUM: find continuous good segments
    segments = find_good_segments(per_frame_good, min_len=seg_min_len, max_gap=seg_max_gap)
    frame_indices = []
    for s, e in segments:
        frame_indices.extend(range(s, e + 1))

    # Always include frame 0
    if 0 not in frame_indices:
        frame_indices = [0] + frame_indices

    return sorted(set(frame_indices))


#  Process one sample 
def process_sample(sample, summary_all, align, args):
    csv_path = os.path.join(CSV_ROOT, f"{sample}_per_frame.csv")
    pt_path  = os.path.join(GVHMR_ROOT, sample, "hmr4d_results.pt")
    if sample not in summary_all:
        print(f"  SKIP {sample}: not in summary.csv"); return None
    if not os.path.exists(csv_path) or not os.path.exists(pt_path):
        print(f"  SKIP {sample}: missing files"); return None

    tier = sample_tier(summary_all[sample],
                       args.fs_contact_high, args.fs_contact_low,
                       args.fs_max_low, args.jerk_max_medium, args.accel_max_medium,
                       args.jerk_low, args.accel_low, args.vel_low)

    csv_data = read_per_frame_csv(csv_path)
    n_frames = len(csv_data["gp_m"])
    good = medium_frame_mask(csv_data, args.gp_t, args.fl_t, args.fs_t, args.jerk_t, args.accel_t, args.vel_t)

    frame_indices = select_frame_indices(
        tier, n_frames, good, args.prefix_frames,
        seg_min_len=args.seg_min_len, seg_max_gap=args.seg_max_gap)

    s = summary_all[sample]
    if tier == "medium":
        segs = find_good_segments(good, args.seg_min_len, args.seg_max_gap)
        seg_str = " segs=" + str([(a,b) for a,b in segs])
    else:
        seg_str = ""
    print(f"  {sample}: tier={tier:6s}  fs_c={float(s['fs_contact_mean_mps']):.3f}"
          f"  jerk={float(s['root_jerk_mps3_max']):.0f}"
          f"  -> {len(frame_indices)} frames{seg_str}")

    data          = torch.load(pt_path, map_location="cpu", weights_only=False)
    params        = data["smpl_params_global"]
    body_pose     = params["body_pose"].numpy()
    global_orient = params["global_orient"].numpy()
    transl        = params["transl"].numpy()

    if global_orient.ndim == 3:
        global_orient = global_orient[:,0,:]
    if body_pose.ndim == 2:
        body_pose = body_pose.reshape(body_pose.shape[0], 21, 3)

    if align:
        global_orient, transl = align_heading(global_orient, transl)

    entry = {"type": "fullbody", "frame_indices": frame_indices,
             "local_joints_rot": [], "root_positions": [], "smooth_root_2d": []}
    for fi in frame_indices:
        full_pose = np.concatenate(
            [global_orient[fi].reshape(1,3), body_pose[fi].reshape(21,3)], axis=0)
        entry["local_joints_rot"].append(full_pose.astype(float).tolist())
        # T-pose pelvis correction (carried from v1 fix)
        root_pos = transl[fi].copy()
        root_pos[1] += SMPL_TPOSE_PELVIS_Y
        entry["root_positions"].append(root_pos.astype(float).tolist())
        entry["smooth_root_2d"].append([float(root_pos[0]), float(root_pos[2])])

    return tier, entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr_root",  required=True, help="dir with <sample>/hmr4d_results.pt (GVHMR --output_root)")
    parser.add_argument("--csv_root",    required=True, help="dir with <sample>_per_frame.csv + summary.csv (../kimodo/eval_physical.py --out_dir)")
    parser.add_argument("--summary_csv", default=None, help="default: <csv_root>/summary.csv")
    parser.add_argument("--out_dir",     required=True, help="where <sample>.json constraint files are written")
    parser.add_argument("--no_align_heading", action="store_true")
    # Sample-level thresholds
    parser.add_argument("--fs_contact_high",  type=float, default=FS_CONTACT_HIGH)
    parser.add_argument("--fs_contact_low",   type=float, default=FS_CONTACT_LOW)
    parser.add_argument("--fs_max_low",       type=float, default=FS_MAX_LOW)
    parser.add_argument("--jerk_max_medium",  type=float, default=JERK_MAX_MEDIUM)
    parser.add_argument("--accel_max_medium", type=float, default=ACCEL_MAX_MEDIUM)
    parser.add_argument("--jerk_low",         type=float, default=JERK_LOW,
                        help="jerk_max above this -> LOW tier (default 3000)")
    parser.add_argument("--accel_low",        type=float, default=ACCEL_LOW,
                        help="accel_max above this -> LOW tier (default 70)")
    parser.add_argument("--vel_low",          type=float, default=VEL_LOW,
                        help="vel_max above this -> LOW tier (default 8)")
    # Per-frame thresholds
    parser.add_argument("--gp_t",   type=float, default=GP_T)
    parser.add_argument("--fl_t",   type=float, default=FL_T)
    parser.add_argument("--fs_t",   type=float, default=FS_T)
    parser.add_argument("--jerk_t",  type=float, default=JERK_T)
    parser.add_argument("--accel_t", type=float, default=ACCEL_T,
                        help="Per-frame root accel threshold m/s2 for MEDIUM (default 15)")
    parser.add_argument("--vel_t",   type=float, default=VEL_T,
                        help="Per-frame root velocity threshold m/s for MEDIUM (default 5)")
    parser.add_argument("--prefix_frames", type=int, default=PREFIX_FRAMES)
    # Segment parameters (v2)
    parser.add_argument("--seg_min_len", type=int, default=5,
                        help="Min frames in a segment to keep (default 3)")
    parser.add_argument("--seg_max_gap", type=int, default=2,
                        help="Max consecutive bad frames to bridge within a segment (default 2)")
    args = parser.parse_args()

    global GVHMR_ROOT, CSV_ROOT, SUMMARY_CSV
    GVHMR_ROOT = args.gvhmr_root
    CSV_ROOT = args.csv_root
    SUMMARY_CSV = args.summary_csv or os.path.join(CSV_ROOT, "summary.csv")

    os.makedirs(args.out_dir, exist_ok=True)
    align = not args.no_align_heading
    summary_all = load_summary(SUMMARY_CSV)
    samples = sorted([d for d in os.listdir(GVHMR_ROOT)
                      if os.path.isdir(os.path.join(GVHMR_ROOT, d))])

    print(f"Processing {len(samples)} samples -> {args.out_dir}\n")
    results = []
    for sample in samples:
        result = process_sample(sample, summary_all, align, args)
        if result is None:
            continue
        tier, entry = result
        out_path = os.path.join(args.out_dir, f"{sample}.json")
        with open(out_path, "w") as f:
            json.dump([entry], f, indent=2)
        results.append({"sample": sample, "tier": tier,
                        "n_frames": len(entry["frame_indices"])})

    # machine-readable tier table for the pipeline driver (step 6 picks the Kimodo duration from it)
    with open(os.path.join(args.out_dir, "tiers.json"), "w") as f:
        json.dump({r["sample"]: r["tier"] for r in results}, f, indent=2)

    print("\n=== Quality Summary ===")
    for r in results:
        bar = {"high": "", "medium": "", "low": ""}[r["tier"]]
        print(f"  {bar}  {r['sample']:<35}  {r['tier']:<6}  {r['n_frames']:3d} frames")


if __name__ == "__main__":
    main()
