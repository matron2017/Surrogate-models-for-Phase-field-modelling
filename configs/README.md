# Configs Cheat Sheet

- `data/` – Phase-field dataset descriptors. Point `PF_DATA_CONFIG` at one of these before running `datapipes/*.sh` (smoke: `phase_field_data_smoke.yaml`).
- `train/` – Active AE-latent training descriptors (AdamW + PSGD variants).
- `visuals/` – Plotting/evaluation descriptors consumed by scripts in `visuals/`.

Usage
- Local run: `python -m models.train.core.train -c configs/train/train_ae_latent_gpumedium_adamw_uncached_freq1_lola_big_64_1024_12g_latent32_nowavelet_b40.yaml`.
- Slurm run: submit one of the active launchers in `slurm/`:
  - `train_ae_latent_gpu_4n16g_lola_big_adamw_latent32_nowavelet_rightclean_fixed34_gradshared_lr1e5.sh`
  - `train_ae_latent_gpu_4n16g_lola_big_adamw_latent32_nowavelet_rightclean_fixed34_gradshared_lr5e6.sh`
  - `train_ae_latent_gpu_4n16g_lola_big_psgd_latent32_nowavelet_rightclean_fixed34_gradshared_lr1e5.sh`
- Datapipes smoke build: `sbatch datapipes/smoke_build.sh` (expects `PF_DATA_CONFIG` to be set).
