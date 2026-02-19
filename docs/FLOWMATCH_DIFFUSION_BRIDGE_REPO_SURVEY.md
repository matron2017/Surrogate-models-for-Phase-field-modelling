# Flow Matching vs Diffusion Bridge (Repo Survey)

Last updated: **2026-02-10**

## Purpose

Background note for ongoing latent-dynamics work: summarize how the local repos frame
flow matching vs diffusion bridge, including:

1. Backbone choices.
2. Model input formulation (noise vs past/source image).
3. Noise process and schedule.
4. Step counts in training/sampling.

## Quick Comparison

| Repo | Method | Backbone(s) | Input framing | Noise process | Typical steps |
|---|---|---|---|---|---|
| `pf_surrogate_modelling` (flow config) | Rectified flow (source-anchored) | `uvit_thermal` | Interpolates between source latent and target latent: `x_t=(1-t)z0+t*y`, with `z0=source` | No explicit Gaussian noising in this objective path | Train samples continuous `t~U(0,1)` |
| `pf_surrogate_modelling` (diffusion config) | Diffusion bridge | `uvit_thermal` | Model sees concatenated `[x_noisy, x_source]` | Brownian-bridge style `x_t=a_t xT + b_t x0 + c_t eps` | Config uses 256 discrete timesteps, one sampled per train step |
| `diffusion_bridge_unified_framework/flow_matching` | Latent rectified flow | SiT transformer (`SiT-B/2` default) | Paired latents `x0->x1`, model predicts velocity on interpolation path | No beta schedule; linear path interpolation | Inference default Euler `steps=50` |
| `diffusion_bridge_unified_framework/diffusion_bridge` | SDE diffusion bridge (GOUB) | Config default `ConditionalUNet` (UNet module; DiT/NAFNet also present) | Conditioned bridge from LQ (`mu`) to GT (`x0`) | Forward noisy state from SDE (`f_mean + f_sigma * noise`) with cosine schedule | Default `T=100` |
| `DBFM-3E8E` | FM + diffusion bridge | FM: SiT; DB: usually ConditionalDiT in options | VAE latent paired LQ/GT setup | UniDB/GOUB-style SDE on DB side, cosine schedule | Commonly `T=100`, `solver_step=20` in options |
| `PBFM` | Physics-based flow matching | Mostly DiT (Darcy can use `Unet3D`) | Starts at Gaussian `x0` and flows toward data `x1` | FM path (no diffusion schedule), optional stochastic sampling perturbation early in rollout | `fm_steps` default `20` |

## Repo-by-Repo Notes

### 1) `pf_surrogate_modelling`

#### Flow matching run path

- Active flow config uses:
  - `train.model_family: flow_matching`
  - `train.objective: rectified_flow_source_anchored`
  - `model.backbone: uvit_thermal`
- In training loop:
  - For source-anchored objective, `z0 = x` (source latent), not Gaussian.
  - `t_match ~ Uniform(0,1)`.
  - `x_t = (1 - t) * z0 + t * y`.
  - target velocity `u_t = y - z0`.

Implication:
- This is not denoising diffusion.
- If source and target latents are close, updates can look like "copying" source.

#### Diffusion-bridge run path

- Active diffusion config uses:
  - `noise_schedule: bridge_linear`
  - `schedule_kwargs.timesteps: 256`
  - `schedule_kwargs.sigma: 1.0`
  - uniform timestep sampler over `[1,255]`
- Bridge forward sample:
  - `x_t = a_t * xT + b_t * x0 + c_t * eps`.
- Model input is concatenated:
  - `x_in = cat([x_noisy, x_source], dim=1)`.

### 2) `diffusion_bridge_unified_framework`

#### Flow matching folder

- Uses SD-VAE latents and SiT backbones.
- Training:
  - encode paired images to latents `(x0, x1)`,
  - sample `t~U(0,1)`,
  - `x_t = t*x1 + (1-t)*x0`,
  - supervise `v = x1 - x0`.
- Inference:
  - Euler integration for chosen number of steps (`--steps`, default 50).

#### Diffusion bridge folder

- Uses SDE classes (GOUB/UniDB style) and denoising models.
- Minimal trainer samples random timestep and noisy state:
  - `x_t = f_mean(x0,t) + f_sigma(t) * noise`,
  - condition is low-quality image/latent `mu`.
- Default minimal config:
  - `T=100`, `schedule=cosine`.

### 3) `DBFM-3E8E`

- Combines both branches:
  - flow matching transformer folder,
  - diffusion bridge transformer folder.
- Flow matching branch matches rectified-flow style latent interpolation + velocity target.
- Diffusion bridge branch (translation/inpainting tasks):
  - encodes GT/LQ to VAE latents,
  - samples noisy states through UniDB/GOUB SDE,
  - trains conditional denoiser with chosen solver settings.
- Typical options:
  - `T: 100`, `schedule: cosine`, `solver_step: 20`.

### 4) `PBFM` (Physics-Based Flow Matching)

- Core FM path:
  - `x0` sampled Gaussian,
  - `x_t = psi_t(x0, x1, t)`,
  - target velocity `u_t(x0,x1)`.
- Adds physics residual terms inside training objective (case-dependent):
  - Darcy residuals,
  - incompressibility/divergence residuals,
  - dynamic-stall physics constraints.
- Optional stochastic sampling perturbation appears in early rollout region during sampling.
- No diffusion beta schedule in main objective.

## Backbone Inventory (from these repos)

1. UNet variants (conditional UNet in diffusion-bridge codepaths).
2. NAFNet variant (available in diffusion-bridge modules).
3. DiT/SiT transformer variants (extensively used in FM branches and some DB options).
4. U-ViT thermal backbone in `pf_surrogate_modelling` (`uvit_thermal`), with bottleneck attention and thermal-field conditioning.

## Practical Interpretation for Current Project

1. Your current `pf_surrogate_modelling` flow setup is source-anchored FM, not Gaussian-start FM.
2. Diffusion-bridge variants explicitly inject noise according to schedule and timestep.
3. If the model appears to copy input in current FM runs, first suspect small source-target displacement and objective framing before only increasing model size.

## Key Code Anchors

- `pf_surrogate_modelling/configs/train/train_flowmatch_uvit_thermal_sourceanchored_latentbest213_gpu5h_b80.yaml`
- `pf_surrogate_modelling/configs/train/train_diffusion_bridge_uvit_thermal_latentbest213_gpu5h_b80.yaml`
- `pf_surrogate_modelling/models/train/core/loops.py`
- `pf_surrogate_modelling/models/train/core/utils.py`
- `pf_surrogate_modelling/models/backbones/uvit_thermal.py`
- `diffusion_bridge_unified_framework/flow_matching/train_vae.py`
- `diffusion_bridge_unified_framework/flow_matching/rectified_flow.py`
- `diffusion_bridge_unified_framework/diffusion_bridge/train.py`
- `diffusion_bridge_unified_framework/diffusion_bridge/models/sde_utils.py`
- `DBFM-3E8E/flow_matching_transformer/train_vae.py`
- `DBFM-3E8E/diffusion_bridge_transformer/diffusion bridge/tasks/translation/options/train.yml`
- `PBFM/darcy_flow/flow_matching.py`
- `PBFM/kolmogorov_flow/flow_matching.py`
- `PBFM/dynamic_stall/flow_matching.py`
