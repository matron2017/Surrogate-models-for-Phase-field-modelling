# Repository Map

This map groups folders by purpose to reduce naming ambiguity.

## Source Code and Config

- `models/`: main Python package (training, backbones, dataloaders).
- `configs/`: YAML/JSON configs (data, train, visuals).
- `slurm/`: Slurm submit scripts (`sbatch` entrypoints).
- `datapipes/`: wrappers for dataset build/preprocessing.
- `scripts/`: project-specific analysis/evaluation scripts.
- `visuals/`: plotting and visualization code/wrappers.
- `tools/`: generic developer utilities (job watcher, tree tools).
- `tests/`: automated tests.
- `experiments/`: prototypes/sandboxes that are not the default production path.
- `docs/`: human documentation and workflow guides.

## Generated Artefacts (Not Core Source)

- `logs/`: runtime logs; `logs/slurm/` stores Slurm stdout/stderr files.
- `results/`: generated figures/CSVs/diagnostics.
- `runs/`: symlink to training run artefacts.
- `runs_debug/`: symlink to debug/smoke run artefacts.
- `mlruns/`: MLflow tracking outputs.

## Naming Rule (Recommended Going Forward)

- Use `*_scripts` or clear README markers if a folder stores launchers, not outputs.
- Keep generated outputs in artefact folders only (`logs/`, `results/`, `runs*`, `mlruns/`).
- Keep reusable code in package/code folders only (`models/`, `visuals/`, `tools/`).
