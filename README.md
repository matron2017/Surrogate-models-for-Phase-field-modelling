# Surrogate Models for Phase-Field Modelling

## Rapid Solidification Modelling Workspace

This directory mirrors only the rapid solidification workflow from `solidification_modelling`:

- `configs/` – YAML for the phase-field dataset, model training, and visuals.
- `datapipes/rapid_solidification/` – VTU → HDF5 builders and wavelet preprocessing scripts.
- `models/` – UNet/U-AFNO backbones plus conditioning modules (with `backbones/`, `conditioning/`, `diffusion/`, `adaptive/` ready for registries).
- `training/core/` – canonical trainer, dataloaders, loss functions, wavelet helpers, and inference utilities.
- `training/slurm/` – ready-to-run Slurm launchers that point at the modules under `training/core/`.
- `visuals/basic/` – lightweight evaluation and inspection tools (with their diagnostics subfolder).
- `visuals/hq/` – high-quality plotting utilities (formerly `visuals_solidification/`).
- `results/` – curated visual outputs and metrics tables produced by the evaluators.
- `logs/` – archived Slurm stdout/stderr for datapipes, model smoke-tests, and training jobs.
- `NEXT_STEPS.md` – checklist of follow-up runs (smoke tests, datapipes, visuals) to
  validate this standalone layout.
- `ENVIRONMENT.md` – Puhti instructions for rebuilding the Tykkÿ/conda container (`physics_ml/`).

Heavy artefacts are re-used via symlinks:

- `data/rapid_solidification -> ../solidification_modelling/data/rapid_solidification`
- `runs -> ../solidification_modelling/runs`
- `runs_debug -> ../solidification_modelling/runs_debug`
- `physics_ml -> ../solidification_modelling/physics_ml`

All training and evaluation launchers now default to the code in this directory. Update the configs or symlinks if you change the workspace root again.
