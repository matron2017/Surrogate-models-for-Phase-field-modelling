# Phase-Field Surrogate Models

Deep-learning surrogate models for 512×512 solidification phase-field PDE simulations.
Trained and evaluated on the [CSC Puhti](https://docs.csc.fi/computing/systems-puhti/) supercomputer (project `project_2008261`).

---

## Overview

Three complementary surrogate model families for predicting the **next PDE frame** from a source frame + thermal field condition:

| Model | Architecture | Params | Description |
|---|---|---|---|
| **Deterministic pixel** | UNet-AFNO + FiLM + ControlNet-XS | 392 M | Pixel-space deterministic predictor |
| **UniDB bridge** | BridgePDEModel + OU bridge SDE | 421 M | Diffusion bridge with cosine schedule |
| **Fractional bridge** | BridgePDEModel + fBm-inspired SDE | 421 M | Smoother bridge paths (Hurst H=0.7) |
| **DC-AE f32c32** | DC-AE encoder–decoder | 323 M | Autoencoder for PDE field compression (32× spatial) |

**Input fields** (3-channel, 512×512 per timestep):
- `φ` — phase field (solid–liquid interface, physical range ≈ [-1, 1])
- `c` — concentration field
- `θ` — thermal field (Kelvin, used as conditioning signal)

---

## Repository structure

```
Phase_field_surrogates/
├── models/                      # Shared model building blocks
│   ├── unet_film_bottleneck.py  # UNet-AFNO with FiLM + ControlNet-XS branch
│   └── mixed_padding.py
├── utils/
│   ├── wavelet_weight.py        # Wavelet-importance-weighted MSE loss
│   ├── data.py / pf_dataloader.py
│   └── config.py
├── deterministic_pixel/         # Pixel-space UNet surrogate
│   ├── scripts/                 # Training script
│   ├── configs*/                # YAML configs (current + archived)
│   └── slurm/                   # Puhti SLURM launchers
├── diffusion_bridge/            # Diffusion bridge models (UniDB + Fractional)
│   ├── models/bridge_wrapper.py
│   ├── sde/                     # UniDB SDE + Fractional bridge SDE
│   ├── scripts/train_bridge.py
│   ├── configs/                 # YAML configs
│   └── slurm/
├── autoencoder_dc_ae/           # DC-AE fine-tuning on PDE data
│   ├── scripts/train_dcae_finetune.py
│   ├── configs_current/         # Active YAML configs (lr2e5, lr5e6 variants)
│   ├── external_refs/DC-Gen/    # Vendored DC-Gen model code (MIT-HAN-Lab)
│   └── slurm/
├── eval/
│   ├── scripts/
│   │   ├── eval_all_models.py   # Multi-sample comparison figures
│   │   └── eval_summary_rows.py # Single-row HQ figures (min/mean/max annotated)
│   └── slurm/                   # Eval SLURM launchers (val + test)
├── external_refs/
│   └── UniDB-plusplus/          # git submodule — UniDB++ reference impl
└── tests/                       # Smoke tests
```

---

## Training

### Deterministic pixel model
```bash
# Puhti — 3 nodes × 4 V100 (12 GPUs total)
sbatch deterministic_pixel/slurm/<launcher>.sh
```

### Diffusion bridge (UniDB or Fractional)
```bash
sbatch diffusion_bridge/slurm/bridge_unidb_big.sh
sbatch diffusion_bridge/slurm/bridge_frac_big.sh
```

**Training loss:**
```
L = MSE(x0_pred, x0_target) + 0.1 × WaveletImportanceMSE(x0_pred, x0_target)
```
The wavelet term up-weights high-frequency / interface-edge errors. The logged `train_mse` tracks plain MSE only.

**Bridge training vs inference:**
- *Training*: random `t ~ U[0,T]`, forward bridge adds noise `x_t = q_sample(x0, x_src, t)`, model predicts clean `x0`
- *Inference*: reverse SDE integration from `x_T = x_src` → `x_0` over `n_steps` (default 20 of T=100)

### DC-AE fine-tuning
```bash
sbatch autoencoder_dc_ae/slurm/dcae_finetune_resume_lr2e5.sh
sbatch autoencoder_dc_ae/slurm/dcae_finetune_resume_lr5e6.sh
```

**Loss:** `L1 + 0.5 × Sobel-gradient (φ channel) + spectral L1 (disabled)`

**LR note:** configs use `reset_lr_on_resume: true` — on resume the cosine schedule restarts from `base_lr` over the remaining epochs (not from the decayed checkpoint value).

---

## Evaluation

```bash
# Full comparison figures (3 samples, val or test data)
sbatch eval/slurm/eval_all_models.sh            # val data
sbatch eval/slurm/eval_all_models_test.sh       # test data  → eval/plots/test/

# Single-row HQ figures with min/mean/max annotations
sbatch eval/slurm/eval_summary_rows.sh          # val data
sbatch eval/slurm/eval_summary_rows_test.sh     # test data  → eval/plots/test/
```

Both eval scripts accept `--h5 <path>` to point at any data file.

---

## Data

Data files are **not tracked in git** (`.gitignore` excludes `*.h5`).

Expected layout under `autoencoder_dc_ae/data/`:
```
train.h5    # ~1.9 GB  — training simulations
val.h5      # ~480 MB  — validation simulations
test.h5     # ~458 MB  — held-out test simulations (10 sims × 253 frames × 512×512)
```

Source data also mirrored at `/scratch/project_2008488/Simon_surrogate/pf_bridge_cleanstack/data/`.

---

## Environment

```bash
# Puhti — activate venv
source /scratch/project_2008261/physics_ml/bin/activate

# Key packages: PyTorch 2.x, h5py, pywavelets, pytorch_wavelets, matplotlib
```

---

## External dependencies

- **UniDB++** (`external_refs/UniDB-plusplus/`) — git submodule from [2769433owo/UniDB-plusplus](https://github.com/2769433owo/UniDB-plusplus)
- **DC-Gen** (`autoencoder_dc_ae/external_refs/DC-Gen/`) — vendored copy of MIT-HAN-Lab DC-AE model code
