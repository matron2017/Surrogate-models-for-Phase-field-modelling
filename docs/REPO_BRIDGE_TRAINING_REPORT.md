# Diffusion bridge training status (repo-facing report)

## Quick status (now)

- Main long job: `31841349` is currently `RUNNING` on `gpu` (`r14g02`), started `2026-02-15T05:12:51Z`.
- Last completed smoke runs: `31846052` passed (15-minute gputest), then completed after a harmless internal step timeout in one helper process.
- `SIGTERM` handling path is added in `train.py` so interrupted runs can finalize status.
- Active config for the long run uses AFNO bottleneck with ConvTranspose2d decoder path.

## Repo orientation (navigation)

Root folders used most often:
- `configs/`: experiment configs
- `scripts/`: custom plotting and watcher helpers
- `slurm/`: queue scripts and launchers
- `logs/slurm/`: scheduler output/error streams
- `runs/`: active/unified run output root from configs
- `results/`, `visuals/`, `models/`, `datapipes/`, `docs/`

Note:
- `runs_debug/` currently exists as a separate folder and is currently empty.
- `runs/` and `runs_debug/` are easy to misread as the same workflow; keep one only for future clarity.
- `runs` is a symlink to `/scratch/project_2008261/solidification_modelling/runs`; keep docs paths consistent before copy/paste operations.

## Active run contract (what is currently active)

- Config: `configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8.yaml`
- Launcher: `slurm/train_unet_bridge_latent_predictnext_12h_1n4g_b64.sh`
- Model family: `unet_bottleneck_attn` with AFNO settings:
  - `afno_depth: 12`
  - `afno_num_blocks: 16`
  - `afno_inp_shape: [8, 8]`
  - `use_bottleneck_attn: false`
- Run contract:
  - `trainer.epochs: 400`
  - `sched.milestones: [200]` (single decay as currently configured for next restart)
  - `sched.gamma: 0.5`
  - `sched.warmup_epochs: 150`
  - `grad_clip: 1.0`
  - `batch_size: 16`
  - AFNO params: `trainable ≈ 452,627,872`

## Next-step guardrails before/after launch

- Before launch:
  - Confirm quota/queue state.
  - Confirm no stale checkpoint watcher duplicate processes.
  - Confirm intended output dir and expected AE dataset path.
- After launch:
  - Track state with `squeue -u $USER`.
  - Tail both `.out` and `.err` for first epoch.
  - Confirm first residual/endpoint metric rows appear and plots are generated.

## Agentic text policy

- In this repo snapshot, a grep for `agentic`/`agentic workflow` returned no hits.
