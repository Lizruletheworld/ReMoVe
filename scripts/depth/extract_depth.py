import sys, os, argparse, glob
import numpy as np
import cv2
import torch

#  stub  DA3  ModuleNotFoundError 
from types import ModuleType as _MT
def _stub(name, **attrs):
    m = _MT(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules.setdefault(name, m)

_stub('moviepy.editor', VideoFileClip=None, concatenate_videoclips=None,
      ImageSequenceClip=None, ColorClip=None, CompositeVideoClip=None)
_stub('gsplat')
_stub('gsplat.rendering')
_stub('pycolmap')
_stub('pillow_heif')
# 

#  DA3  
from depth_anything_3.api import DepthAnything3

#  CLI  
parser = argparse.ArgumentParser(description="Batch Video Depth Extraction using DA3")
parser.add_argument("--video_dir",  required=True, help="Directory of mp4 videos, or a single mp4 file")
parser.add_argument("--out_dir",    default="outputs/depth", help="Output directory")
parser.add_argument("--batch",      type=int, default=8, help="Batch size for DA3 inference")
# 
parser.add_argument("--gpu_id",     type=int, default=0, help="Current GPU ID to use")
parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs being used")
args = parser.parse_args()

VIDEO_DIR   = args.video_dir
OUT_DIR     = args.out_dir
DA3_BATCH   = args.batch
OUTPUT_FPS  = 24

# (set http_proxy/https_proxy in the environment if the HF download needs a proxy)

os.makedirs(OUT_DIR, exist_ok=True)


def load_video_bgr(path):
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    return frames, fps


def write_video(path, frames_bgr, fps):
    h, w = frames_bgr[0].shape[:2]
    tmp_path = path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    os.rename(tmp_path, path)


def main():
    # 1.  mp4 
    if os.path.isfile(VIDEO_DIR):
        video_paths = [VIDEO_DIR]
    else:
        video_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + glob.glob(os.path.join(VIDEO_DIR, "*.MP4"))
    video_paths = sorted(list(set(video_paths)))  # 
    
    total_videos = len(video_paths)
    if total_videos == 0:
        print(f"[ERROR] No .mp4 files found in directory: {VIDEO_DIR}")
        return

    #   
    #  gpu_id 
    my_video_paths = [path for i, path in enumerate(video_paths) if i % args.world_size == args.gpu_id]
    print(f"[1] GPU {args.gpu_id}/{args.world_size}: Found {len(my_video_paths)}/{total_videos} videos to process.")
    
    if not my_video_paths:
        print(f"[INFO] GPU {args.gpu_id} has no tasks assigned. Exiting.")
        return
    # 

    # 2.  DA3 
    print(f"[2] GPU {args.gpu_id} Loading DA3METRIC-LARGE ...")
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model = model.to(device).eval()
    print(f"    Model loaded on {device}\n" + "="*50)

    # 3. 
    for idx, video_path in enumerate(my_video_paths, 1):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_depth_path = os.path.join(OUT_DIR, f"{stem}_depth.mp4")
        out_npz_path = os.path.join(OUT_DIR, f"{stem}_depth_maps.npz")
        
        #   
        if os.path.exists(out_depth_path) and os.path.exists(out_npz_path):
            print(f"  [GPU {args.gpu_id}] [{idx}/{len(my_video_paths)}] Skipped (Already processed): {os.path.basename(video_path)}")
            continue
        # 
        
        print(f"\n [GPU {args.gpu_id}] [{idx}/{len(my_video_paths)}] Processing: {os.path.basename(video_path)}")
        
        # 
        frames, fps = load_video_bgr(video_path)
        T = len(frames)
        if T == 0:
            print(f"    [WARN] No frames found in {video_path}, skipping.")
            continue
        H, W = frames[0].shape[:2]
        print(f"    Total frames={T} | Size={W}x{H} | Original FPS={fps:.2f}")

        # 
        print("    Running DA3 inference ...")
        depth_maps = []
        
        for start in range(0, T, DA3_BATCH):
            batch_bgr = frames[start:start + DA3_BATCH]
            batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]
            
            with torch.no_grad():
                pred = model.inference(batch_rgb)
                
            for d in pred.depth:
                d_resized = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                depth_maps.append(d_resized)
                
            print(f"    [{start + len(batch_bgr)}/{T}] frames processed", flush=True)

        depth_maps = np.stack(depth_maps, axis=0)  # (T, H, W)
        print(f"    Depth range: {depth_maps.min():.2f}m ~ {depth_maps.max():.2f}m")

        #  metric 
        print(f"    Saving raw depth maps to NPZ  {out_npz_path}")
        np.savez_compressed(out_npz_path, depth=depth_maps)

        #  Turbo 
        print(f"    Writing depth visualization video  {out_depth_path}")
        d_min, d_max = depth_maps.min(), depth_maps.max()
        depth_vis = []
        
        for d in depth_maps:
            norm = ((d - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
            depth_vis.append(colored)
            
        write_video(out_depth_path, depth_vis, fps)

    print(f"\n[ GPU {args.gpu_id} ALL DONE ]")


if __name__ == "__main__":
    main()