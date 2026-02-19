# Models Package Overview

This directory holds all learnable components plus the training entrypoint.

## Start Here

- Training entrypoint: `models/train/core/train.py`
- Experiment configs: `configs/train/*.yaml`
- Backbone registry: `models/backbones/registry.py`

## Common Model Paths

- Backbones: `models/backbones/` (active: UViT variants + retained compatibility backbones)
- Latent autoencoder: `models/latent/`
- Latent rectified-flow backbone: `models/latent_rf_unet_controlnet.py`
- Diffusion/flow helpers: `models/diffusion/`
- Training loops + config plumbing: `models/train/`

## Conditioning Conventions

- Active training path does **not** use scalar conditioning.
- Dataset-level scalar-conditioning flags (`return_cond`, `conditioning.use_cond`) are rejected in active trainer setup.
- Thermal-field conditioning is spatial: set `conditioning.use_theta: true` and provide the thermal channels as field tensors.

## Design Rationale (Current Focus)

- Latent-space flow/diffusion: compress 1024x1024 fields to 64x64 latents for faster training and larger backbones.
- U-ViT baseline: U-Net shape with FiLM-style feature modulation and a single attention bottleneck for a stable latent-flow reference.
- AFNO bottleneck variants: swap the bottleneck mixer to AFNO for global spectral mixing with lower attention cost.
- DiT-style backbones are not present in this repo; see `diffusion_bridge_unified_framework` for the external reference implementation.
- Rectified/flow-matching objective: predict velocity along linear paths for stable, efficient training on dynamics.
- Thermal/ControlNet path: spatial thermal conditioning is handled by the latent RF ControlNet-style backbone when needed.

## Minimal Random-Data Smoke

Run a tiny forward + loss check on CPU:

```bash
PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3 - <<'PY'
import torch
from models.latent_rf_unet_controlnet import LatentRFUNetControlNet

model = LatentRFUNetControlNet(Cz=4, channels=[8, 16, 16, 16], blocks_per_level=1, dropout=0.0, use_attn_bottleneck=False)
b, cz, h, w = 2, 4, 16, 16
z_n = torch.randn(b, cz, h, w)
z_np1 = torch.randn(b, cz, h, w)
t = torch.rand(b, 1)
z0 = torch.randn_like(z_np1)
z_t = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * z_np1
u_star = z_np1 - z0
theta = torch.randn(b, 1, h, w)
v = model(z_t, t, z_n, theta)
loss = (v - u_star).pow(2).mean()
print("shape", v.shape, "loss", float(loss))
PY
```
