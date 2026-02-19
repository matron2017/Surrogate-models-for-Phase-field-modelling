# Training Core Layout

This directory is the runtime entry path for training/evaluation (`models.train.core.train`).

## Main modules

- `train.py`: top-level orchestration (config parsing, setup, epoch loop, checkpointing, logging).
- `loops.py`: train/validation epoch loops and objective-specific forward logic.
- `setup.py`: model/dataloader/optimizer/scheduler construction.
- `pf_dataloader.py`: paired phase-field dataset + thermal-field loading and normalization.
- `logging.py`: CSV + console logging helpers.
- `optim_sched.py`: optimizer/scheduler factories.
- `loss_functions.py`: objective-specific loss terms.
- `utils.py`: shared utility primitives (batch prep, distributed reductions, schedule sampling kernels).
- `latent.py`: latent encode/split/select helpers for AE-backed training.
- `diffusion_forward.py`: diffusion-family noisy-state sampling + model-input assembly helpers.
- `metric_stats.py`: channel-wise accumulators and VRMSE reduction helpers.

## Design intent

- Keep `train.py` and `loops.py` focused on control flow.
- Keep math/schedule details in small focused helpers (`diffusion_forward.py`, `metric_stats.py`, `utils.py`).
- Keep data/conditioning behavior inside dataloader + batch prep (`pf_dataloader.py`, `utils.py`).

## Safety for queued jobs

- Public call sites used by SLURM scripts are unchanged (`-m models.train.core.train`).
- The refactor only moved helper internals; training configs and launch scripts do not need updates.
