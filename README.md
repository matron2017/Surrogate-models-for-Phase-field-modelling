# Surrogate Models for Phase-Field Modelling

Research codebase for latent surrogate modelling of phase-field rapid solidification.
The active training stack supports autoencoder (AE), diffusion, and flow-matching backbones.

## Repository Layout

- `models/` - training code, backbones, losses, and dataloaders.
- `configs/` - experiment YAML files.
- `slurm/` - example job submission scripts.
- `scripts/` - utilities for preprocessing, diagnostics, and evaluation.
- `visuals/` - plotting and figure-generation helpers.
- `tests/` - regression and smoke tests.
- `docs/README.md` - high-level documentation index.

Generated artifacts are intentionally separated from source:
- `logs/`, `results/`, `mlruns/`, `runs*`, and datasets/checkpoints.

## Core Training Entrypoint

All training launchers call:

```bash
python -m models.train.core.train -c <config.yaml>
```

## Typical Workflow

1. Validate repository state with `git status --short --branch`.
2. Run a short smoke job first.
3. Launch longer training after smoke checks pass.
4. Monitor jobs and collect metrics with scripts in `scripts/`.

## Included Documentation

Only repository-level documentation intended for general reuse is tracked in Git.
System-specific operational notes are intentionally kept out of GitHub.

## Contributing

See `CONTRIBUTING.md`.
