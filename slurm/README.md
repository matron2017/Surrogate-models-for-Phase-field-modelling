# Slurm Launchers

This folder contains `sbatch` submit scripts (launchers), not job output.

- Submit from here when starting training jobs.
- Prefer `gputest` smoke launchers first, then long `gpu` runs.
- Each launcher points to a config under `configs/train/`.
- Job stdout/stderr is written to `logs/slurm/`.
- Keep launcher edits and matching config edits in the same commit when possible.
- Before submitting bridge runs, run the preflight + error-scan pair in this repo:
  - `python scripts/bridge_preflight_check.py --config <config.yaml>`
  - `python scripts/report_bridge_errors.py --logs-dir logs/slurm --hours 72 --min-severity ERROR`
- Repository artifact roots are intentionally centralized via `pf_surrogate_modelling/runs` and `pf_surrogate_modelling/runs_debug`.
  - Both are symlinks to:
    - `/scratch/project_2008261/solidification_modelling/runs`
    - `/scratch/project_2008261/solidification_modelling/runs_debug`

If you need runtime logs, go to `logs/slurm/` instead.
