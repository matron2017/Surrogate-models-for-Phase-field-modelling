# Surrogate Models for Phase-Field Modelling

Research codebase for latent surrogate modelling of phase-field rapid solidification.
The active training stack supports autoencoder (AE), diffusion, and flow-matching backbones
with Slurm launchers for CSC systems.

## Repository Layout

- `models/` - training code, backbones, losses, and dataloaders.
- `configs/` - experiment YAML files.
- `slurm/` - job submission scripts.
- `scripts/` - project utilities and diagnostics.
- `visuals/` - evaluation and plotting tools.
- `tests/` - CPU-oriented regression tests.
- `docs/` - workflow, environment, and cluster notes.

Generated artifacts are intentionally separated from source:
- `logs/`, `results/`, `mlruns/`, `runs*`.

## Core Training Entrypoint

All training launchers call:

```bash
python -m models.train.core.train -c <config.yaml>
```

## Quick Start (Puhti)

1. Check repository state:
   - `git status --short --branch`
2. Run a smoke launcher from `slurm/` first (`gputest` partition).
3. Move to longer runs on `gpu` only after smoke success.
4. Monitor with `squeue -u $USER` and `tail -f logs/slurm/<job>.out`.

## LUMI-G Scaling Workflow

Use these files for LUMI migration and scaling:

- `docs/LUMI_BENCHMARK_PREP_2026.md` - migration checklist and benchmark scope.
- `docs/LUMI_EXTREME_SCALE_SCALING_PLAN.md` - benchmark matrix and reporting contract.
- `slurm/lumi_g_ae_smoke.sh` - AE portability/sanity run.
- `slurm/lumi_g_backbone_scaling_job.sh` - main strong/weak scaling job script.
- `slurm/lumi_g_submit_backbone_scaling_matrix.sh` - batch submit helper for matrix runs.

## Contributing and Workflow

- `CONTRIBUTING.md` - contribution expectations.
- `docs/GIT_WORKFLOW.md` - practical staging/testing workflow.
- `docs/START_HERE.md` - onboarding order for this repository.
