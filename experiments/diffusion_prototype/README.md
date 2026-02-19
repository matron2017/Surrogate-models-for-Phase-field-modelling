# Rapid Solidification Diffusion Playground

This folder gathers the Hugging Face `diffusers` prototype requested for latent
phase-field modelling. It complements the existing surrogate workflow without
interfering with the production trainers in `rapid_solidification/`.

## Layout

```
experiments/diffusion_prototype/
  data/                  # placeholder for NPZ/HDF5 dumps (tracked via .gitkeep)
  src/
    dataset.py           # lightweight HDF5 dataset helper
    train_vae.py         # AutoencoderKL training loop
    train_diffusion.py   # latent diffusion trainer
    make_placeholder_data.py # utility to generate dummy HDF5 tensors
    models_vae.py        # Autoencoder builders
    models_unet.py       # UNet builders
    wavelet_loss.py      # shared wavelet regulariser
  configs/
    vae_config.json      # defaults for VAE training
    unet_config.json     # defaults for diffusion training
```

Drop any raw `.npy`/`.h5` bundles inside `data/` (or point the configs to the
existing `/scratch/project_2008261/rapid_solidification/data/...` assets).

- Residual DDPM/AFNO scalar-conditioning components were removed as part of the
  thermal-field-only cleanup. This prototype folder now focuses on the
  `train_vae.py` + `train_diffusion.py` pathway.

## Environment & Dependencies

1. Use the Tykkÿ container shipped in `/scratch/project_2008261/physics_ml`.
2. Install HF deps (already run once):
   ```bash
   /scratch/project_2008261/physics_ml/bin/pip3 install --upgrade \
     'diffusers[torch]' transformers accelerate
   ```
   The command writes into `~/.local`; Python automatically picks it up.
3. (Optional) clone HF diffusers for reference/examples:
   ```bash
   git clone https://github.com/huggingface/diffusers.git /scratch/project_2008261/diffusers_ref
   /scratch/project_2008261/physics_ml/bin/pip3 install -e /scratch/project_2008261/diffusers_ref"[torch]"
   ```
4. Submit Slurm jobs with the GPU-friendly partitions only (`gputest` for smoke,
   `gpu` for longer runs) because CPU quotas on `small`/`medium` are exhausted.
5. Add completion alerts to new SBATCH scripts via:
   ```bash
   #SBATCH --mail-type=END,FAIL
   #SBATCH --mail-user=<you>@csc.fi
   ```
   or wrap submissions with `squeue --wait -j <jobid>` to block until exit.

More container maintenance details live in `docs/ENVIRONMENT.md`.

## Placeholder Dataset (2×1024×1024 + Scalars)

For quick smoke tests without the real PF HDF5, generate a dummy file with the
same shapes (2-channel grids + thermal/time scalars per frame):

```bash
cd /scratch/project_2008261/rapid_solidification
/scratch/project_2008261/physics_ml/bin/python3.11 \
  experiments/diffusion_prototype/src/make_placeholder_data.py \
  --frames 2 \
  --output experiments/diffusion_prototype/data/placeholder_smoke.h5
```

The default configs already point at this file. Each sequence contains:

- `images`: `[N, 2, 1024, 1024]` phase fields.
- `scalars`: `[N, 2]` with columns `[thermal_gradient, normalized_time]`.

When `pair_mode=true`, the dataloader returns dictionaries with `current`,
`next`, and `scalars` (the scalars correspond to the `current` frame and are
broadcast inside the diffusion trainer as extra conditioning channels).

## Baseline vs “v2” Settings

| Component | Baseline (configs/*) | Scaled “v2” idea |
|-----------|----------------------|------------------|
| VAE latent channels | 4 | 8 |
| VAE block widths | (256, 512, 512, 512) | (320, 640, 960, 960) |
| UNet blocks | (320, 640, 960, 960) | (512, 768, 1024, 1024) + extra attn |
| Scheduler | 1000 DDPM steps, linear beta | 2000 steps, cosine beta |
| Wavelet weight | 0.1 (VAE) / 0.0 (diffusion) | 0.2 / 0.1 |
| Target params | ~80 M | 200–300 M |

Start with the baseline to verify IO + training, then scale per column 2 once
profiling looks healthy.

## Running the Trainers

### VAE
```bash
cd /scratch/project_2008261/rapid_solidification
PY=/scratch/project_2008261/physics_ml/bin/python3.11
$PY experiments/diffusion_prototype/src/train_vae.py \\
  --config experiments/diffusion_prototype/configs/vae_config.json
```
Override paths/batch size in the config for custom dumps.

### Diffusion (latent UNet)
```bash
cd /scratch/project_2008261/rapid_solidification
PY=/scratch/project_2008261/physics_ml/bin/python3.11
$PY experiments/diffusion_prototype/src/train_diffusion.py \\
  --config experiments/diffusion_prototype/configs/unet_config.json
```
Ensure `training.vae_checkpoint` points at a trained VAE weight file before
kick-off.

## Next Steps

- Wire these scripts into Slurm launchers (`gputest` for smoke jobs) with
  `--mail-type` alerts so completions are pushed automatically.
- Extend configs with Optuna-ready parameter sweeps once the baseline is stable.
- Port the wavelet weights and dataset sampling tricks from the legacy trainer
  once quality metrics are comparable.
