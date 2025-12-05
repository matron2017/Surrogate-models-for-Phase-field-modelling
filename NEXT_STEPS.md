# Rapid Solidification — Diffusion-Oriented Next Steps

Focus all work inside `rapid_solidification/`. Tackle the following sequence rather than attempting the full refactor in one pass.

---

## 1. Align configs with explicit experiment descriptors
- Introduce the descriptor fields (`train.model_family`, `model.backbone`, `diffusion.noise_schedule`, `diffusion.timestep_sampler`, `loss.weight_wavelet_loss`, `adaptive.region_selector`, `adaptive.enable_adaptive_resolution`) into the existing YAML configs under `configs/train_model/rapid_solidification/`.
- Thread these fields through the current training entrypoint `train_solidification/train.py` (or equivalent) so they are read once and passed around without ad-hoc parsing.
- Ensure descriptors are plumbed to MLflow logging (params + tags). Gate the MLflow connection off `MLFLOW_TRACKING_URI` for Puhti compatibility.

## 2. Introduce lightweight registries
- ✅ `models/registry.py`, `models/diffusion/scheduler_registry.py`, `models/diffusion/timestep_sampler.py`, and `models/adaptive/registry.py` exist, and `training/core/train.py` now falls back to them when configs omit legacy `model.file` entries.
- ✅ `train_solidification/loss_registry.py` provides surrogate/diffusion loss builders and is wired into the trainer; extend it with edge-aware wrappers as they land.
- Extend the registries with additional backbones (SSM/SineNet), diffusion schedule variants, and adaptive selectors once the corresponding implementations exist.

## 3. Refactor the trainer into a thin orchestrator
- Trainer now pulls descriptors up front, builds models via the registry, instantiates loss builders, and prepares noise schedules + timestep samplers when `model_family == "diffusion"`.
- Diffusion forward pass (`q_sample`, sampler-driven timesteps, region-selector hook) is wired into the training branch; validation for diffusion and the downstream MLflow logging are the remaining pieces.
- Keep backbone-specific or adaptive logic out of the trainer; registries should continue to supply specialised behaviour.

## 4. Harden MLflow + Slurm usage
- ✅ Trainer now reads `MLFLOW_TRACKING_URI`, sets/starts runs on rank 0, logs descriptor params + flattened config, streams epoch metrics, and uploads CSV/plots/checkpoints as artefacts.
- TODO: wire in parent-run tracking for resume workflows and ensure diffusion validation metrics feed into MLflow once the eval path is in place.

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
