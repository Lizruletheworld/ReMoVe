import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image
from models.wan_remove import load_pipeline, load_remove_weights, encode_video, condition_hook

NEGATIVE_PROMPT = "blurry, overexposed, static, low detail, subtitles, watermark, distorted anatomy, extra limbs, malformed hands, deformed face"
DEFAULT_LORA = os.path.join(os.environ.get("REMOVE_ROOT", "."), "weights", "wan_lora", "Wan2.2-TI2V-5B-remove_step-31110.safetensors")


def load_frames(path, height, width):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError("No frames: " + path)
    return frames


def fit_frames(frames, count):
    frames = frames[:count]
    while len(frames) < count:
        frames.append(frames[-1])
    return frames


def save_video(frames, path, fps):
    import imageio.v2 as imageio
    arr = [np.asarray(frame).astype(np.uint8) for frame in frames]
    imageio.mimsave(path, arr, fps=fps, codec="libx264", quality=8)


def main():
    ap = argparse.ArgumentParser(description="ReMoVe inference with official Wan2.2 Diffusers.")
    ap.add_argument("--input_image", required=True)
    ap.add_argument("--mesh_video", required=True)
    ap.add_argument("--depth_video", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_LORA)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_frames", default="auto")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    image = Image.open(args.input_image).convert("RGB")
    width = image.width - image.width % 16
    height = image.height - image.height % 16
    image = image.resize((width, height))
    mesh = load_frames(args.mesh_video, height, width)
    source_len = len(mesh)
    count = ((source_len - 1) // 4) * 4 + 1 if args.num_frames == "auto" else int(args.num_frames)
    if (count - 1) % 4:
        raise ValueError("num_frames must be 4k+1")
    mesh = fit_frames(mesh, count)
    depth = fit_frames(load_frames(args.depth_video, height, width), count)

    pipe = load_pipeline(device, dtype)
    load_remove_weights(pipe.transformer, args.ckpt)
    mesh_latent = encode_video(pipe, mesh, device, dtype)
    depth_latent = encode_video(pipe, depth, device, dtype)
    handle = pipe.transformer.patch_embedding.register_forward_pre_hook(condition_hook(mesh_latent, depth_latent))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        result = pipe(
            image=image,
            prompt=args.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=count,
            generator=generator,
        )
    finally:
        handle.remove()
    frames = result.frames[0] if isinstance(result.frames, list) and result.frames and isinstance(result.frames[0], list) else result.frames
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "remove_wan.mp4")
    save_video(frames, out, args.fps)
    print(out)


if __name__ == "__main__":
    main()
