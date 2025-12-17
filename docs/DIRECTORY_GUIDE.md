# Directory Guide

- models/: Python package with backbones, diffusion/flow-matching logic, and train scripts.
- configs/: YAML configs for training/visuals. New flow-matching and diffusion smokes live under configs/train/.
- slurm/: Slurm launchers; DDP helpers for gputest (1-node/4-GPU, 2-node/8-GPU).
- docs/: Centralised guides (dev, env, monitoring, checklists).
- data/: Rapid solidification HDF5/JSON assets (unchanged).
- logs/: Slurm stdout/err.
- runs_debug/: Outputs from smoke runs (metrics, plots, checkpoints).
- visuals/, tools/, datapipes/: Utilities for preprocessing and plotting.
- experiments/: Prototypes (e.g., diffusion_prototype).

All imports assume `models/` as the package root; do not rename the package.
