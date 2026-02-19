# Thermal-Field Conditioning Audit (2026-02-11)

This note documents exactly how thermal conditioning is wired in this workspace, what exists outside `pf_surrogate_modelling`, and the layer/channel/spatial sizes for the key models.

## Scope

- Repository focus: `pf_surrogate_modelling`
- External references checked: `DBFM-3E8E`, `DiffusionBridge`, `RDBM`, `PBFM`, `ControlNet-XS`
- Runtime focus: current pending 7h jobs
  - `train_diffusion_bridge_unet_thermal_latentbest213_gpu2h_1gpu_8x8_212m.yaml`
  - `train_diffusion_bridge_unet_thermal_latentbest213_gpu2h_1gpu_8x8_212m_rdbmres.yaml`

## Current 7h Job Backbone + Config Delta

- Both pending 7h jobs use `unet_bottleneck_attn` (`UNetFiLMAttn`) with latent input/output:
  - effective model input path: `x_model = [x_noisy, x_source, theta]`
  - output channels: `32`
- Base vs RDBM-res config difference:
  - `*_rdbmres.yaml` adds:
    - `residual_mode: abs`
    - `residual_normalize: true`
    - `residual_scale: 1.0`
    - `residual_power: 1.0`
    - `residual_clip: null`
    - `residual_eps: 1.0e-6`
  - backbone size/channels are the same.

## Thermal Injection Paths Found In `pf_surrogate_modelling`

1. Active UNet diffusion path (used by pending 7h jobs): early spatial concat.
   - Thermal map loaded from data and appended in dataset:
     - `models/train/core/pf_dataloader.py` (`thermal_field` / `thermal_images`)
   - Split as `theta`, normalized, then concatenated to UNet input:
     - `models/train/core/utils.py`
     - `models/train/core/loops.py`
   - No dedicated thermal branch parameters inside the UNet.

2. U-ViT thermal path (`uvit_thermal`): dedicated thermal branch + per-stage modulation.
   - Thermal pyramid encoder: `ThermalPyramidMapper`
   - Per-stage modulation convs: `theta_enc_mod`, `theta_dec_mod`
   - Optional FiLM/additive modulation mode.

3. ControlNet-style latent RF model (available in repo, not currently selected by configs):
   - `models/latent_rf_unet_controlnet.py`
   - Dedicated thermal control encoder + zero-conv injection.

## External Conditioning Resources (Outside `pf_surrogate_modelling`)

1. `DBFM-3E8E`
   - Bridge UNet/DiT conditioning is image-based concat style:
     - `x = xt - cond; x = cat([x, cond])`
   - No thermal-specific branch in this codebase.

2. `DiffusionBridge`
   - DDBM UNet supports `condition_mode == "concat"`:
     - doubles input channels and concatenates `xT`.
   - Generic bridge conditioning, not thermal-specific.

3. `RDBM`
   - Conditioned bridge with `mu` endpoint and residual-noise scheduling.
   - Conditioning is not a separate thermal branch.

4. `PBFM`
   - Gradient/condition signal support in UNet3D via concat and attention.
   - Can inspire richer condition pathways, but not directly thermal-field-specific.

5. `ControlNet-XS`
   - Full ControlNet paradigm exists (hint branch + control params), generic for image hints.
   - Useful reference if stronger control branch is needed in PF surrogate.

## Exact Model Size + Shape Audit

The following values come from live model instantiation + forward hooks at latent resolution `64x64`.

### A) Active Diffusion UNet (`UNetFiLMAttn`, current 7h jobs)

- Total trainable params: `204,514,208`
- Model input channels: `65` (`64` from `[x_noisy, x_source]` + `1` theta)
- Model output channels: `32`
- Thermal branch params: `0` (thermal is concat-only in this path)

Layer output shapes:

| Layer | Output shape |
|---|---|
| `in_conv` | `(1, 192, 64, 64)` |
| `enc_blocks.0.0` | `(1, 192, 64, 64)` |
| `enc_blocks.0.1` | `(1, 192, 64, 64)` |
| `downs.0` | `(1, 384, 32, 32)` |
| `enc_blocks.1.0` | `(1, 384, 32, 32)` |
| `enc_blocks.1.1` | `(1, 384, 32, 32)` |
| `downs.1` | `(1, 768, 16, 16)` |
| `enc_blocks.2.0` | `(1, 768, 16, 16)` |
| `enc_blocks.2.1` | `(1, 768, 16, 16)` |
| `downs.2` | `(1, 1024, 8, 8)` |
| `enc_blocks.3.0` | `(1, 1024, 8, 8)` |
| `enc_blocks.3.1` | `(1, 1024, 8, 8)` |
| `bot_pre.0` | `(1, 1024, 8, 8)` |
| `bot_pre.1` | `(1, 1024, 8, 8)` |
| `bot_attn` | `(1, 1024, 8, 8)` |
| `bot_post.0` | `(1, 1024, 8, 8)` |
| `bot_post.1` | `(1, 1024, 8, 8)` |
| `up_convs.0` | `(1, 768, 16, 16)` |
| `dec_blocks.0.0` | `(1, 768, 16, 16)` |
| `dec_blocks.0.1` | `(1, 768, 16, 16)` |
| `up_convs.1` | `(1, 384, 32, 32)` |
| `dec_blocks.1.0` | `(1, 384, 32, 32)` |
| `dec_blocks.1.1` | `(1, 384, 32, 32)` |
| `up_convs.2` | `(1, 192, 64, 64)` |
| `dec_blocks.2.0` | `(1, 192, 64, 64)` |
| `dec_blocks.2.1` | `(1, 192, 64, 64)` |
| `out_conv` | `(1, 32, 64, 64)` |

### B) `UVitThermalSurrogate` (thermal branch model in registry)

- Total trainable params: `1,213,600`
- Thermal-branch params: `257,984`
  - `theta_mapper`: `197,760`
  - `theta_ctx`: `20,736`
  - `theta_enc_mod`: `29,056`
  - `theta_dec_mod`: `10,432`
- Input channels: `64`, output channels: `32`

Layer output shapes:

| Layer | Output shape |
|---|---|
| `theta_mapper.stem` | `(1, 64, 64, 64)` |
| `theta_mapper.downs.0` | `(1, 64, 32, 32)` |
| `theta_mapper.downs.1` | `(1, 64, 16, 16)` |
| `in_conv` | `(1, 32, 64, 64)` |
| `theta_enc_mod.0` | `(1, 64, 64, 64)` |
| `enc_blocks.0` | `(1, 32, 64, 64)` |
| `downs.0` | `(1, 64, 32, 32)` |
| `theta_enc_mod.1` | `(1, 128, 32, 32)` |
| `enc_blocks.1` | `(1, 64, 32, 32)` |
| `downs.1` | `(1, 96, 16, 16)` |
| `theta_enc_mod.2` | `(1, 192, 16, 16)` |
| `enc_blocks.2` | `(1, 96, 16, 16)` |
| `attn` | `(1, 96, 16, 16)` |
| `ups.0` | `(1, 64, 32, 32)` |
| `theta_dec_mod.0` | `(1, 128, 32, 32)` |
| `ups.1` | `(1, 32, 64, 64)` |
| `theta_dec_mod.1` | `(1, 64, 64, 64)` |
| `out_conv` | `(1, 32, 64, 64)` |

### C) `LatentRFUNetControlNet` (ControlNet-style thermal branch)

- Total trainable params: `39,180,704`
- Thermal-branch params: `5,378,560`
  - `theta_encoder`: `5,164,672`
  - `theta_zero_convs`: `213,888`

Layer output shapes:

| Layer | Output shape |
|---|---|
| `stem` | `(1, 128, 64, 64)` |
| `theta_encoder.blocks.0` | `(1, 128, 64, 64)` |
| `theta_encoder.downs.0` | `(1, 256, 32, 32)` |
| `theta_encoder.blocks.1` | `(1, 256, 32, 32)` |
| `theta_encoder.downs.1` | `(1, 256, 16, 16)` |
| `theta_encoder.blocks.2` | `(1, 256, 16, 16)` |
| `theta_encoder.downs.2` | `(1, 256, 8, 8)` |
| `theta_encoder.blocks.3` | `(1, 256, 8, 8)` |
| `theta_zero_convs.0` | `(1, 128, 64, 64)` |
| `enc_blocks.0.0` | `(1, 128, 64, 64)` |
| `enc_blocks.0.1` | `(1, 128, 64, 64)` |
| `enc_blocks.0.2` | `(1, 128, 64, 64)` |
| `enc_blocks.0.3` | `(1, 128, 64, 64)` |
| `downs.0` | `(1, 256, 32, 32)` |
| `theta_zero_convs.1` | `(1, 256, 32, 32)` |
| `enc_blocks.1.0` | `(1, 256, 32, 32)` |
| `enc_blocks.1.1` | `(1, 256, 32, 32)` |
| `enc_blocks.1.2` | `(1, 256, 32, 32)` |
| `enc_blocks.1.3` | `(1, 256, 32, 32)` |
| `downs.1` | `(1, 256, 16, 16)` |
| `theta_zero_convs.2` | `(1, 256, 16, 16)` |
| `enc_blocks.2.0` | `(1, 256, 16, 16)` |
| `enc_blocks.2.1` | `(1, 256, 16, 16)` |
| `enc_blocks.2.2` | `(1, 256, 16, 16)` |
| `enc_blocks.2.3` | `(1, 256, 16, 16)` |
| `downs.2` | `(1, 256, 8, 8)` |
| `theta_zero_convs.3` | `(1, 256, 8, 8)` |
| `enc_blocks.3.0` | `(1, 256, 8, 8)` |
| `enc_blocks.3.1` | `(1, 256, 8, 8)` |
| `enc_blocks.3.2` | `(1, 256, 8, 8)` |
| `enc_blocks.3.3` | `(1, 256, 8, 8)` |
| `attn` | `(1, 256, 8, 8)` |
| `ups.0` | `(1, 256, 16, 16)` |
| `dec_blocks.0.0` | `(1, 256, 16, 16)` |
| `dec_blocks.0.1` | `(1, 256, 16, 16)` |
| `dec_blocks.0.2` | `(1, 256, 16, 16)` |
| `dec_blocks.0.3` | `(1, 256, 16, 16)` |
| `ups.1` | `(1, 256, 32, 32)` |
| `dec_blocks.1.0` | `(1, 256, 32, 32)` |
| `dec_blocks.1.1` | `(1, 256, 32, 32)` |
| `dec_blocks.1.2` | `(1, 256, 32, 32)` |
| `dec_blocks.1.3` | `(1, 256, 32, 32)` |
| `ups.2` | `(1, 128, 64, 64)` |
| `dec_blocks.2.0` | `(1, 128, 64, 64)` |
| `dec_blocks.2.1` | `(1, 128, 64, 64)` |
| `dec_blocks.2.2` | `(1, 128, 64, 64)` |
| `dec_blocks.2.3` | `(1, 128, 64, 64)` |
| `out_conv` | `(1, 32, 64, 64)` |

## Practical Rationale

1. Active 7h UNet path is high-capacity and simple:
   - best for immediate training continuity and speed.
   - thermal enters as an aligned spatial field channel.

2. `uvit_thermal` gives explicit per-stage thermal modulation:
   - low parameter cost thermal branch.
   - more ControlNet-like than pure input concat, but still compact.

3. `latent_rf_unet_controlnet` is the closest ControlNet pattern:
   - dedicated thermal encoder + zero-conv residual injection.
   - stronger conditioning control at higher parameter cost.

