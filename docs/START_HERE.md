# Start Here (First 15 Minutes)

If this is your first time in this repository, follow this order:

1. Check repo state first:
   - `git status --short --branch`
2. Read `README.md` for the project overview and active training launchers.
3. Read `docs/REPO_MAP.md` to understand which folders are source code vs generated artefacts.
4. Read `docs/GIT_WORKFLOW.md` for safe staging/testing habits in this HPC workspace.
5. If jobs are pending/running, read `docs/REPO_REORG_PLAN_2026-02-17.md` before moving or renaming files.
6. Run a quick local validation:
   - `PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests/test_backbones_rs.py -q`
7. Inspect active Slurm launchers in `slurm/` (these are submit scripts, not logs).
8. Monitor live job output in `logs/slurm/` (these are generated stdout/stderr files).

## Current Active Workflow

1. Build/prepare data with `datapipes/` as needed.
2. Train AE latent model using one launcher in `slurm/`.
3. Check progress in `logs/slurm/*.out`.
4. Evaluate or visualize using `scripts/` and `visuals/`.
5. Before starting a new bridge run:
   - `python scripts/bridge_preflight_check.py --config <config.yaml>`
   - `python scripts/report_bridge_errors.py --logs-dir logs/slurm --hours 48 --min-severity ERROR`
   - Confirm no unresolved `B-xxx` entry in `docs/ERROR_TRACKER.md` for the target run path.
   - One-command alternative:
     - `python scripts/check_bridge_readiness.py --config <config.yaml> --hours 48 --min-severity WARN`

If any step reports issues, fix first, then rerun the preflight checks before submitting again.

## Common Confusions

- `slurm/` vs `logs/slurm/`:
  - `slurm/` contains job submission scripts (`sbatch ...`).
  - `logs/slurm/` contains outputs produced by those jobs.
- `scripts/` vs `tools/` vs `visuals/`:
  - `scripts/`: project-specific analysis/eval jobs.
  - `tools/`: developer/utilitarian helpers (watchers, tree tools).
  - `visuals/`: plotting/evaluation modules and wrappers.
- `results/` vs `runs_debug/` vs `mlruns/`:
  - `results/`: generated analysis outputs.
  - `runs_debug/`: training artefacts/checkpoints (symlinked workspace).
  - `mlruns/`: MLflow run metadata/artifacts.
