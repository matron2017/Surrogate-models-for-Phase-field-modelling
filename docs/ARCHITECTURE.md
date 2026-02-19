# Rapid Solidification Repo Layout (LEAN)

This document records the trimmed directory layout after removing legacy code paths.
Code sits under `models/`; launchers live in `slurm/`; generated artefacts are isolated in output directories.

```
configs/                 # YAML descriptors for data, training, visuals
datapipes/               # sbatch/CLI wrappers invoking package datapipes
docs/                    # Dev guide, environment notes, monitoring, checklists
experiments/             # Prototypes (e.g., diffusion_prototype)
models/                  # Python package (backbones, conditioning, diffusion, train/*)
slurm/                   # Batch launchers pointing to models/train/core/train.py
scripts/                 # Project-specific analysis scripts
tools/                   # Generic helper utilities
tests/                   # Automated tests
visuals/                 # Plotting/evaluation modules and wrappers
logs/, results/, runs_debug/, mlruns/  # Generated artefacts/outputs
README.md                # Quickstart & directory map
ARCHITECTURE.md          # (this file)
```

Key distinction:
- `slurm/` = submit scripts.
- `logs/slurm/` = generated outputs from submitted jobs.

Artefacts under `logs/`, `results/`, `runs/`, `runs_debug/`, and `mlruns/` should remain output-only. Keep source code in `models/`, `scripts/`, `visuals/`, `tools/`, and `tests/`.
