import argparse
import numpy as np


def trim_motion(npz_path, out_path=None, fps=30, threshold_ratio=0.3, tail_pad=5):
    d = np.load(npz_path)
    keys = list(d.keys())

    if 'posed_joints' not in d:
        print(f"Key 'posed_joints' not found. Available keys: {keys}")
        return

    joints = d['posed_joints']  # (L, J, 3)
    L = joints.shape[0]
    print(f"Input: {L} frames ({L/fps:.1f}s)")

    # Per-frame velocity = mean L2 norm of joint displacements
    vel = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=-1)  # (L-1,)
    threshold = vel.mean() * threshold_ratio

    active = np.where(vel > threshold)[0]
    if len(active) == 0:
        print("No active frames found  keeping all frames.")
        end_frame = L
    else:
        last_active = active[-1] + 1  # +1 because vel[i] = diff between frame i and i+1
        end_frame = min(last_active + tail_pad + 1, L)

    print(f"Active frames: {active[0] if len(active) else 'none'}  {active[-1] if len(active) else 'none'}")
    print(f"Trimming to {end_frame} frames ({end_frame/fps:.1f}s)")

    trimmed = {}
    for k in keys:
        arr = d[k]
        if hasattr(arr, '__len__') and len(arr) == L:
            trimmed[k] = arr[:end_frame]
        else:
            trimmed[k] = arr

    if out_path is None:
        stem = npz_path.replace('.npz', '')
        out_path = f"{stem}_trimmed.npz"

    np.savez_compressed(out_path, **trimmed)
    print(f"Saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input npz file from Kimodo')
    parser.add_argument('--out', default=None, help='Output path (default: input_trimmed.npz)')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--threshold_ratio', type=float, default=0.3,
                        help='Frames with velocity < mean*ratio are considered static')
    parser.add_argument('--tail_pad', type=int, default=5,
                        help='Extra frames to keep after last active frame')
    args = parser.parse_args()

    trim_motion(args.input, out_path=args.out, fps=args.fps,
                threshold_ratio=args.threshold_ratio, tail_pad=args.tail_pad)


if __name__ == '__main__':
    main()
