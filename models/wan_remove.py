import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model

WAN_BASE = os.environ.get("WAN_BASE_MODEL") or (os.path.join(os.environ.get("REMOVE_WEIGHTS", "weights"), "wan_base") if os.path.isdir(os.path.join(os.environ.get("REMOVE_WEIGHTS", "weights"), "wan_base")) else "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
LORA_RANK = 32
LORA_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "net.0.proj", "net.2"]


def expand_patch_embedding_for_remove(transformer):
    old = transformer.patch_embedding
    cin, cout = old.in_channels, old.out_channels
    new = nn.Conv3d(
        cin * 3, cout, kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, dilation=old.dilation, groups=old.groups,
        bias=old.bias is not None,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight[:, :cin].copy_(old.weight)
        new.weight[:, cin:].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)
    transformer.patch_embedding = new
    if hasattr(transformer, "in_dim"):
        transformer.in_dim = cin * 3
    return transformer


def condition_hook(mesh_latent, depth_latent):
    def hook(module, args):
        noisy = args[0]
        mesh = mesh_latent.expand(noisy.shape[0], -1, -1, -1, -1)
        depth = depth_latent.expand(noisy.shape[0], -1, -1, -1, -1)
        size = noisy.shape[2:]
        if mesh.shape[2:] != size:
            mesh = F.interpolate(mesh.float(), size=size, mode="trilinear", align_corners=False).to(noisy.dtype)
        if depth.shape[2:] != size:
            depth = F.interpolate(depth.float(), size=size, mode="trilinear", align_corners=False).to(noisy.dtype)
        return (torch.cat([noisy, mesh, depth], dim=1),)
    return hook


def load_pipeline(device, dtype=torch.bfloat16):
    from diffusers import WanImageToVideoPipeline
    pipe = WanImageToVideoPipeline.from_pretrained(WAN_BASE, torch_dtype=dtype)
    expand_patch_embedding_for_remove(pipe.transformer)
    lora = LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK, target_modules=LORA_MODULES)
    inject_adapter_in_model(lora, pipe.transformer)
    pipe.to(device)
    pipe.vae.to(device=device, dtype=torch.float32)
    return pipe


def load_remove_weights(transformer, path):
    state = load_file(path)
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    print(f"Loaded ReMoVe weights: missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected


def encode_video(pipe, frames, device, dtype):
    if hasattr(pipe, "video_processor") and hasattr(pipe.video_processor, "preprocess_video"):
        tensor = pipe.video_processor.preprocess_video(frames)
    else:
        array = np.stack([np.asarray(frame.convert("RGB")) for frame in frames]).astype(np.float32)
        array = array / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0)
    with torch.no_grad():
        encoded = pipe.vae.encode(tensor.to(device=device, dtype=torch.float32))
        latent = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded.latents
    mean = torch.tensor(pipe.vae.config.latents_mean, device=device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    std = (1.0 / torch.tensor(pipe.vae.config.latents_std, device=device, dtype=latent.dtype)).view(1, -1, 1, 1, 1)
    return ((latent - mean) * std).to(device=device, dtype=dtype)
