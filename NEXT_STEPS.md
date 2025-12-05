# Rapid Solidification — Diffusion-Oriented Next Steps

Focus all work inside `rapid_solidification/`. Tackle the following sequence rather than attempting the full refactor in one pass.

---

## 1. Align configs with explicit experiment descriptors
- Introduce the descriptor fields (`train.model_family`, `model.backbone`, `diffusion.noise_schedule`, `diffusion.timestep_sampler`, `loss.weight_wavelet_loss`, `adaptive.region_selector`, `adaptive.enable_adaptive_resolution`) into the existing YAML configs under `configs/train_model/rapid_solidification/`.
- Thread these fields through the current training entrypoint `train_solidification/train.py` (or equivalent) so they are read once and passed around without ad-hoc parsing.
- Ensure descriptors are plumbed to MLflow logging (params + tags). Gate the MLflow connection off `MLFLOW_TRACKING_URI` for Puhti compatibility.

## 2. Introduce lightweight registries
- Create `models/registry.py` (or extend it if present) exposing `build_model(model_family, backbone, model_cfg)` and an internal backbone registry that maps `"unet"`, `"uafno"`, `"ssm"`, `"sinenet"`, etc. to constructors.
- Add `models/diffusion/scheduler_registry.py` for noise schedules plus a sibling module for timestep samplers (`get_noise_schedule`, `get_timestep_sampler`). Start with `"linear"`, `"cosine"`, `"logsnr_laplace"`, `"uniform"`, `"logsnr_importance"`, and reserve `"adaptive_region"`.
- Under `train_solidification/`, add `loss_registry.py` implementing `build_diffusion_loss` with hooks for `weight_wavelet_loss` and placeholder `edge_loss`.
- Stub `models/adaptive/registry.py` with `build_region_selector` returning `"none"` (identity) today, leaving quadtree / edge-mask hooks for later.

## 3. Refactor the trainer into a thin orchestrator
- Restructure `train_solidification/train.py` so it:
  - builds datasets/dataloaders from the canonical PF datapipe (single source of truth for `batch = {"input", "target", "cond", "meta"}`),
  - calls the registries to build the model, diffusion schedule, sampler, loss, and region selector,
  - runs one shared training loop with a minimal branch on `model_family` (surrogate vs diffusion).
- Keep backbone-specific or adaptive logic out of the trainer; registries should supply all specialised behaviour.
- Wire optional region-selector hooks into the diffusion branch even if they currently no-op.

## 4. Harden MLflow + Slurm usage
- Read `MLFLOW_TRACKING_URI` from the environment and call `mlflow.set_tracking_uri` early; default to project storage on Puhti (e.g., `file:/scratch/$PROJECT/mlruns`).
- Use `config["mlflow"]["experiment_name"]` (or a fallback) and `SLURM_JOB_ID` as the run name; ensure only rank 0 interacts with MLflow in DDP.
- Log the descriptor fields, flattened config, and the main metrics (train/val loss, `metric.val_rel_l2`, physics metrics). Save checkpoints/plots as artefacts.
- Add the optional parent-run mechanism (store `mlflow_parent_run_id` in checkpoints, set `mlflow.parentRunId` on resume).

## 5. Keep Optuna + Slurm orchestration consistent
- Ensure the Optuna driver populates the new descriptor fields before launching `sbatch`.
- Update worker scripts (e.g., `training/slurm/*.sh` + their Python entrypoints) so every trial:
  - writes sampled hyperparameters into the shared config schema,
  - runs `train_solidification/train.py`,
  - pushes the objective (`metric.val_rel_l2`) to both Optuna and MLflow using identical names.

## 6. Establish the minimal feedback loop
- Add/refresh CPU unit tests:
  - ✅ `tests/test_pf_dataloader_rs.py` now covers the PF dataloader contract using a synthetic HDF5 stub; extend it as loaders grow optional features.
  - `tests/test_backbones_rs.py` for the registered backbones on small inputs.
  - `tests/test_diffusion_schedulers.py` for schedules + samplers (monotone β, reasonable log-SNR, sampler distribution sanity).
  - `tests/test_train_one_step_rs.py` to run a tiny config for a couple of optimiser steps and check loss decreases.
- Document a debug training command (e.g., `python -m train_solidification.train --config configs/train_model/rapid_solidification/diffusion/debug_cosine.yaml`) plus an evaluation/plotting command (e.g., `python -m visuals.run_dataset_eval_and_plots --config configs/eval/rapid_solidification/diffusion_standard.yaml --checkpoint <path>`).
- List smoke/CI commands in `AGENTS.md` (or equivalent): `pytest -q`, the debug training run, and the evaluation command.

## 7. Leave the door open for adaptive / region-aware methods
- ✅ Flattened `models/` so `backbones/`, `conditioning/`, `diffusion/`, and `adaptive/` are first-class directories ready for registries.
- Keep backbone code under `models/backbones/`, diffusion utilities under `models/diffusion/`, and adaptive hooks under `models/adaptive/`.
- Avoid hard-coding spatial sizes; pass dataset-derived shapes into models/losses so quadtree or patch-based diffusion can drop in later.
- Make sure descriptor fields already include region/adaptive toggles (`region_selector`, `adaptive_resolution`) so future QDM-style quadtree or adaptive-timestep approaches can be compared cleanly in MLflow.

Document progress in this file as milestones complete, and keep future TODOs scoped to `rapid_solidification/` unless explicitly re-scoped later.
