# Directory Guide

- models/: Main Python package (backbones, latent models, trainer, dataloaders).
- configs/: Train/data/visual configuration files.
- slurm/: Job launchers (`sbatch` scripts); does not store runtime logs.
- logs/: Generated runtime logs; `logs/slurm/` stores Slurm stdout/stderr.
- results/: Generated diagnostics/plots/CSVs.
- runs/ and runs_debug/: Symlinked training artefact roots.
- mlruns/: MLflow run metadata/artifacts.
- scripts/: Project-specific analysis and dataset utility scripts.
- visuals/: Reusable plotting/evaluation code and wrappers.
- tools/: Generic developer helpers (job watcher, tree/tools).
- experiments/: Prototype/sandbox workflows outside the mainline path.
- docs/: Human documentation and onboarding maps.
- tests/: Automated checks for active code paths.
- datapipes/: Dataset conversion/build wrappers.

All imports assume `models/` as the package root; do not rename the package.
