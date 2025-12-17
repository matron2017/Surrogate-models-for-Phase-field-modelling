# Rapid Solidification Repo Layout (LEAN)

This document records the trimmed directory layout after removing legacy log/run artefacts. Code now sits under the `models/` Python package; Slurm/shell wrappers stay at repo root.

```
configs/                 # YAML descriptors for data, training, visuals
datapipes/               # sbatch/CLI wrappers invoking package datapipes
docs/                    # Dev guide, environment notes, monitoring, checklists
experiments/             # Prototypes (e.g., diffusion_prototype)
models/                  # Python package (backbones, conditioning, diffusion, train/*)
slurm/                   # Batch launchers pointing to models/train/core/train.py
visuals/                 # Shell wrappers for plotting
logs/, runs_debug/, results/  # Artefacts and outputs (ignored in git)
README.md                # Quickstart & directory map
ARCHITECTURE.md          # (this file)
```

Artefacts under `logs/`, `runs/`, `runs_debug/`, and `mlruns/` are ignored by git; avoid checking in heavy outputs. Expect future runs to create fresh `runs_debug/…` paths under `/scratch` when configs/training jobs execute. Keep new artefacts (checkpoints, MLflow outputs, Slurm logs) under a dedicated directory if needed, e.g., `artifacts/`.
