## Rapid Solidification — Current Experiment Log

Status snapshot date: **February 9, 2026**.

### Slurm jobs (AE latent)
- **31701755** `ae_latent_gpu_4n16g_lola_big_adamw_latent32_rc_lr1e5`: `CANCELLED` (manual stop to save billing units)
  - Launcher: `slurm/train_ae_latent_gpu_4n16g_lola_big_adamw_latent32_nowavelet_rightclean_fixed34_gradshared_lr1e5.sh`
  - Config: `configs/train/train_ae_latent_gpumedium_adamw_uncached_freq1_lola_big_64_1024_12g_latent32_nowavelet_b40.yaml`
  - Last saved checkpoint epoch: `213`
- **31701756** `ae_latent_gpu_4n16g_lola_big_adamw_latent32_rc_lr5e6`: `CANCELLED` (manual stop to save billing units)
  - Launcher: `slurm/train_ae_latent_gpu_4n16g_lola_big_adamw_latent32_nowavelet_rightclean_fixed34_gradshared_lr5e6.sh`
  - Config: `configs/train/train_ae_latent_gpumedium_adamw_uncached_freq1_lola_big_64_1024_12g_latent32_nowavelet_b40_lr5e6.yaml`
  - Last saved checkpoint epoch: `53`
- **31701757** `ae_latent_gpu_4n16g_lola_big_psgd_latent32_rc_lr1e5`: `RUNNING`
  - Launcher: `slurm/train_ae_latent_gpu_4n16g_lola_big_psgd_latent32_nowavelet_rightclean_fixed34_gradshared_lr1e5.sh`
  - Config: `configs/train/train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_latent32_nowavelet_b40.yaml`

### Current workflow scope
- Active training path is AE latent-space training with thermal-field-ready data plumbing.
- Legacy scalar-conditioning wrappers and legacy smoke launchers were removed from `slurm/`.
- Use `models/train/core/train.py` + one of the active configs in `configs/train/`.

### Archived training snapshot (preserved logs + convergence)
- Snapshot root: `results/training_history/2026-02-09_ae_adamw_psgd_snapshot/`
- Raw copied logs: `results/training_history/2026-02-09_ae_adamw_psgd_snapshot/raw_logs/`
- Compressed archive: `results/training_history/2026-02-09_ae_adamw_psgd_snapshot/slurm_logs_snapshot.tar.gz`
- Per-epoch convergence CSVs: `results/training_history/2026-02-09_ae_adamw_psgd_snapshot/*_epochs.csv`
- Summary + resume checkpoint paths: `results/training_history/2026-02-09_ae_adamw_psgd_snapshot/SUMMARY.md`

### Immediate action items
1. Continue monitoring job `31701757` for PSGD stability and metric spikes.
2. Resume AdamW runs only if needed, using checkpoint paths documented in the snapshot summary.
3. Keep storing future run snapshots under `results/training_history/` before cleaning `logs/slurm/`.
