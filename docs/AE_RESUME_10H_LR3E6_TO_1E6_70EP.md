# AE Resume Plan (10h, +70 epochs, LR 3e-6 -> 1e-6)

This run continues the current PSGD latent AE training from the latest saved epoch while preserving the prior best checkpoint separately.

## Launcher

- Slurm script:
  `pf_surrogate_modelling/slurm/train_ae_latent_gpu_4n16g_lola_big_psgd_latent32_resume10h_lr3e6_to1e6_70ep.sh`

## Behavior

- Source checkpoint bundle:
  `pf_surrogate_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel`
- Reads latest `checkpoint.last.pth` at job start.
- Copies current `checkpoint.best.pth` and `checkpoint.last.pth` to protected backups in the new run folder.
- Creates a resume checkpoint that:
  - keeps model (and optimizer moments) from latest state,
  - resets scheduler state,
  - sets optimizer lr to `3e-6`.
- Generates a temporary config at job start with:
  - `trainer.epochs = source_epoch + 70`,
  - `sched.name = cosine`,
  - `sched.T_max = 70`,
  - `sched.eta_min = 1e-6`,
  - output to a new run directory (no overwrite of the old run).

## Outputs

- New run out_dir:
  `pf_surrogate_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128_resume10h_lr3e6_to1e6_70ep`
- Bootstrap artifacts:
  - `bootstrap_from_prev_run/checkpoint.best.pre_resume_copy.pth`
  - `bootstrap_from_prev_run/checkpoint.last.pre_resume_copy.pth`
  - `bootstrap_from_prev_run/checkpoint.resume_lr3e6_reset_sched.pth`
  - `bootstrap_from_prev_run/resume_manifest.json`
