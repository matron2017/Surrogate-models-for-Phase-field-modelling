# AE Capacity Justification (Current LoLA-Style Run)

Last updated: **2026-02-10**

## Question

Is the current AE architecture too small, or is it already big enough for this project stage?

Short answer: for the current rightclean dataset and budget constraints, it is reasonable to treat the current AE as **already large enough** and focus effort on optimizer/training stability instead of adding more layers.

## Compared Architectures

### Current training architecture (this repo)

- Config: `configs/train/train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_latent32_nowavelet_b40.yaml`
- Model class: `models/latent/ae_lola_model.py` (`LatentAELoLAModel`)
- Core AE: `models/latent/autoencoder_lola.py` (`LoLAAutoencoder2D`)
- Width/depth: `hid_channels=[64,128,256,512,768]`, `hid_blocks=[3,3,3,3,3]`
- Latent channels: `32`

### LoLA large reference architecture

- Config: `../lola_upstream/experiments/configs/ae/dcae_f32c64_large.yaml`
- Same LoLA DCAE-style blocks (`DCEncoder`/`DCDecoder`)
- Width/depth: `hid_channels=[64,128,256,512,768,1024]`, `hid_blocks=[3,3,3,3,3,3]`
- Latent channels: `64`

## Trainable Parameters

- Current AE total trainable params: **142,471,906** (~142.47M)
- LoLA large total trainable params: **313,095,170** (~313.10M)
- Ratio (current / LoLA-large): **0.455x** (45.5%)

Interpretation:
- Current model is not "small" in absolute terms.
- The gap to LoLA-large is mainly one extra top stage (`1024` width) rather than fundamentally different lower/mid hierarchy.

## Stage-by-Stage Parameter Comparison

Values below are encoder stage + mirrored decoder stage totals.

| Stage | Width | Blocks | Current AE | LoLA Large |
|---|---:|---:|---:|---:|
| 0 | 64 | 3 | 445,506 | 445,506 |
| 1 | 128 | 3 | 2,361,216 | 2,361,216 |
| 2 | 256 | 3 | 9,441,024 | 9,441,024 |
| 3 | 512 | 3 | 37,756,416 | 37,756,416 |
| 4 | 768 | 3 | 92,467,744 | 92,024,576 |
| 5 | 1024 | 3 | N/A | 171,066,432 |

What this means:
- Up through stage 3 (64->512), current and LoLA-large are effectively the same capacity.
- Current AE also includes the 768 stage.
- The major missing piece vs LoLA-large is only the extra 1024 stage (very expensive in params).

## Dataset Scale Context

From the active rightclean HDF5s:

- Train set: `data/stochastic/rightclean/simulation_train_rightclean_fixed34_gradshared.h5`
  - groups: `40`
  - frames: `13,225`
  - `pairs_idx`: `13,185`
- Val set: `data/stochastic/rightclean/simulation_val_rightclean_fixed34_gradshared.h5`
  - groups: `10`
  - frames: `3,355`
  - `pairs_idx`: `3,345`

Given this data scale, the practical bottleneck is more likely optimization behavior (learning-rate regime, preconditioner dynamics, checkpoint selection, early stopping policy) than lack of raw AE parameter count.

## Decision for Current Budget Window

Recommended default:

1. Keep the current AE architecture fixed for now.
2. Prioritize optimizer/schedule/selection decisions (PSGD vs AdamW dynamics, val metric selection, run length).
3. Only upscale architecture if underfitting evidence appears in both train and val (not just noisy validation spikes).

## When to Revisit "Model Too Small"

Re-open architecture scaling if all are true:

1. Train metrics plateau early at unsatisfactory values despite stable optimization.
2. Validation also plateaus at the same level (no signs of recoverable optimization issue).
3. Longer training + LR/scheduler/precondition tuning fails to improve.

If those conditions hold, next step is adding the 1024 stage (LoLA-large-like), accepting significantly higher compute/memory cost.
