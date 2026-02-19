# Open Questions + Resources (Thermal-Field Conditioning)

## Confirmed Current State

- **No DiT in `pf_surrogate_modelling`**. The DiT implementation only exists in
  `diffusion_bridge_unified_framework` and is not wired into the training stack here.
- **Thermal-field-only conditioning is the intended default** for current diffusion/flow
  experiments; scalar thermal-gradient conditioning is not used.
- **+1000 Euler-step pairing is handled in the data pipeline** (no doc action needed here).

## Resources (paths)

- **Best trained AE checkpoint**
  - `pf_surrogate_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent64_wavelet_multiband_beta150_multistep110_b40/LatentAELoLAModel/checkpoint.best.pth`
- **Thermal-field conditioning backbone (ControlNet-style)**
  - `pf_surrogate_modelling/models/latent_rf_unet_controlnet.py`
- **Thermal-field loader behavior**
  - `pf_surrogate_modelling/models/train/core/pf_dataloader.py`
- **Latent diffusion/flow configs**
  - `configs/train/train_flowmatch_*_latent*.yaml`
  - `configs/train/train_diffusion_*_latent*.yaml`
- **External DiT reference (not wired here)**
  - `diffusion_bridge_unified_framework/diffusion_bridge/models/modules/DenoisingDiT_arch.py`
- **External PBFM reference (cloned locally)**
  - `/scratch/project_2008261/PBFM`

## Open Questions (with action plan)

### 1) Thermal-field autoencoder?

- **Option A**: Keep full-res thermal field and downsample to latent resolution by interpolation only.
- **Option B**: Train a thermal AE and inject thermal latents instead of full-res maps.
- **Decision criteria**: memory cost vs. forcing fidelity.

### 2) U-ViT conditioning with thermal field

- **Default**: ControlNet-style theta injection (thermal map injected into intermediate layers).
- **Alternative (non-default)**: Input-channel concat (documented fallback only).
