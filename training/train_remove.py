import argparse
import json
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from models.wan_remove import expand_patch_embedding_for_remove, load_remove_weights, WAN_BASE, LORA_MODULES, LORA_RANK
from peft import LoraConfig, inject_adapter_in_model


class MeshDepthDataset(Dataset):
    def __init__(self, metadata_path, height, width, frames):
        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = records.get("data", records.get("samples", []))
        if not records:
            raise ValueError("metadata contains no samples")
        self.records = records
        self.base_dir = Path(metadata_path).resolve().parent
        self.height = height
        self.width = width
        self.frames = frames

    def __len__(self):
        return len(self.records)

    def _read(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path
        cap = cv2.VideoCapture(str(path))
        out = []
        while len(out) < self.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            out.append(frame)
        cap.release()
        if not out:
            raise RuntimeError("empty video: " + path)
        while len(out) < self.frames:
            out.append(out[-1])
        x = np.stack(out[:self.frames]).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(x).permute(3, 0, 1, 2)

    def __getitem__(self, index):
        rec = self.records[index]
        video = self._read(os.path.expanduser(rec["video_path"]))
        mesh = self._read(os.path.expanduser(rec["mesh_path"]))
        depth = self._read(os.path.expanduser(rec["depth_path"]))
        prompt = rec.get("motion_caption") or rec.get("caption") or rec.get("prompt") or ""
        return {"video": video, "mesh": mesh, "depth": depth, "prompt": prompt}


def parse_args():
    p = argparse.ArgumentParser(description="Train ReMoVe mesh/depth conditioning on official Wan2.2.")
    p.add_argument("--metadata_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default=WAN_BASE)
    p.add_argument("--init_checkpoint", default=None)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    return p.parse_args()


def encode_video(vae, video, device, dtype):
    video = video.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        encoded = vae.encode(video)
        latent = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded.latents
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    std = (1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=latent.dtype)).view(1, -1, 1, 1, 1)
    return ((latent - mean) * std).to(dtype=dtype)


def timestep_sigmas(scheduler, timesteps, device, dtype):
    schedule = scheduler.timesteps.to(device)
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    indices = [(schedule == t).nonzero(as_tuple=False).item() for t in timesteps]
    shape = [len(indices)] + [1] * 4
    return sigmas[indices].view(*shape)


def save_adapter(accelerator, transformer, output_dir, step):
    from safetensors.torch import save_file
    model = accelerator.unwrap_model(transformer)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
             if "patch_embedding" in k or "lora_" in k}
    path = Path(output_dir) / ("checkpoint-" + str(step) + ".safetensors")
    save_file(state, str(path), metadata={"format": "pt", "step": str(step)})
    accelerator.print("saved " + str(path))


def main():
    args = parse_args()
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must be 4k+1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from accelerate import Accelerator
    from diffusers import WanImageToVideoPipeline
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16")
    device = accelerator.device
    dtype = torch.bfloat16

    pipe = WanImageToVideoPipeline.from_pretrained(args.model, torch_dtype=dtype)
    transformer = pipe.transformer
    expand_patch_embedding_for_remove(transformer)
    lora = LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK, target_modules=LORA_MODULES)
    inject_adapter_in_model(lora, transformer)
    if args.init_checkpoint:
        load_remove_weights(transformer, args.init_checkpoint)
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    for name, parameter in transformer.named_parameters():
        if "patch_embedding" in name or "lora_" in name:
            parameter.requires_grad = True
    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    transformer.train()
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    if getattr(pipe, "image_encoder", None) is not None:
        pipe.image_encoder.requires_grad_(False)
        pipe.image_encoder.eval()
    pipe.vae.to(device)
    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.text_encoder.to(device)
    transformer.to(device)

    dataset = MeshDepthDataset(args.metadata_path, args.height, args.width, args.num_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)
    pipe.transformer = transformer
    scheduler = pipe.scheduler
    scheduler.set_timesteps(1000, device=device)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(loader, disable=not accelerator.is_local_main_process, desc="epoch " + str(epoch))
        for batch in progress:
            with accelerator.accumulate(transformer):
                video = encode_video(pipe.vae, batch["video"], device, dtype)
                mesh = encode_video(pipe.vae, batch["mesh"], device, dtype)
                depth = encode_video(pipe.vae, batch["depth"], device, dtype)
                first_frame = encode_video(pipe.vae, batch["video"][:, :, :1], device, dtype)
                bsz = video.shape[0]
                timesteps = scheduler.timesteps[torch.randint(0, len(scheduler.timesteps), (bsz,), device=device)]
                sigmas = timestep_sigmas(scheduler, timesteps, device, video.dtype)
                noise = torch.randn_like(video)
                noisy = (1.0 - sigmas) * video + sigmas * noise
                if getattr(pipe.config, "expand_timesteps", False):
                    first_frame_mask = torch.ones_like(video[:, :1])
                    first_frame_mask[:, :, 0] = 0
                    latent_input = (1.0 - first_frame_mask) * first_frame + first_frame_mask * noisy
                    patch = transformer.config.patch_size
                    timestep_input = (
                        first_frame_mask[0, 0, :, ::patch[1], ::patch[2]]
                        * timesteps[:, None, None, None]
                    ).flatten(1)
                else:
                    latent_input = noisy
                    timestep_input = timesteps.to(device=device)
                full = torch.cat([latent_input, mesh, depth], dim=1)
                with torch.no_grad():
                    prompt_embeds, _ = pipe.encode_prompt(
                        prompt=list(batch["prompt"]),
                        negative_prompt=None,
                        do_classifier_free_guidance=False,
                        device=device,
                        dtype=dtype,
                    )
                output = transformer(
                    hidden_states=full,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                target = noise - video
                loss = F.mse_loss(output.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                progress.set_postfix(loss=f"{loss.item():.5f}", step=step)
                if step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(args.output_dir, exist_ok=True)
                        save_adapter(accelerator, transformer, args.output_dir, step)
                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_adapter(accelerator, transformer, args.output_dir, step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
