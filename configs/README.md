# Configs Cheat Sheet

- `data/` – Phase-field dataset descriptors. Point `PF_DATA_CONFIG` at one of these before running `datapipes/*.sh` (smoke: `phase_field_data_smoke.yaml`).
- `train/` – Training descriptors for UNet/UAFNO/FNO, wavelet variants, and diffusion tasks. Files ending in `_smoke.yaml` are ≤15 min on `gputest`; larger configs target long `gpu` runs.
- `visuals/` – Plotting/evaluation descriptors consumed by scripts in `visuals/`.

Usage
- Local run: `python -m models.train.core.train -c configs/train/train_smoke.yaml`.
- Slurm smoke: `sbatch slurm/train_smoke.sh` (reads a config in `configs/train/`).
- Datapipes smoke build: `sbatch datapipes/smoke_build.sh` (expects `PF_DATA_CONFIG` to be set).
