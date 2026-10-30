#!/usr/bin/env python3
import argparse
import os

import torch
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video


MODEL_ID = os.environ.get("WAN_BASE_MODEL") or (os.path.join(os.environ.get("REMOVE_WEIGHTS", "weights"), "wan_base") if os.path.isdir(os.path.join(os.environ.get("REMOVE_WEIGHTS", "weights"), "wan_base")) else "Wan-AI/Wan2.2-TI2V-5B-Diffusers")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a coarse video with the official Wan2.2 TI2V-5B model."
    )
    parser.add_argument("--image", required=True, help="Path to the first-frame image.")
    parser.add_argument("--prompt", required=True, help="Text prompt for Wan2.2.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Wan2.2 TI2V-5B Diffusers model or local model directory.",
    )
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be divisible by 16.")
    if args.num_frames < 1:
        raise ValueError("--num_frames must be positive.")

    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
    dtype = torch.bfloat16
    pipe = WanImageToVideoPipeline.from_pretrained(args.model, torch_dtype=dtype)

    if args.device.startswith("cuda"):
        pipe.to(args.device)
        pipe.vae.to(dtype=torch.float32)
    else:
        pipe.to(args.device)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    result = pipe(
        image=image,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    frames = result.frames[0] if isinstance(result.frames[0], list) else result.frames
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    export_to_video(frames, output, fps=args.fps)
    print(f"Coarse video saved to: {output}")


if __name__ == "__main__":
    main()
