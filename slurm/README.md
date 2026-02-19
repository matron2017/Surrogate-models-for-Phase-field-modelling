# Slurm Launchers

This folder contains `sbatch` launcher scripts (not output logs).

## General Use

- Submit from repository root so relative paths resolve.
- Run smoke tests first, then longer runs.
- Keep launcher and config edits in the same commit.
- Job logs are written to `logs/slurm/`.

## LUMI-G Scripts

- `lumi_g_ae_smoke.sh`
  - AE portability smoke on LUMI-G.
  - Use `RUN_TASKS=1/2/4/8` for single-node GPU-count checks.

- `lumi_g_backbone_scaling_job.sh`
  - Main scaling launcher for latent backbone workloads.
  - `MODE=strong` keeps global batch fixed.
  - `MODE=weak` keeps per-GPU batch fixed.

- `lumi_g_submit_backbone_scaling_matrix.sh`
  - Submits strong and weak scaling node grids.
  - Defaults to `1 2 4 8` nodes and chooses `small-g` or `standard-g` by node count.

- `lumi_g_scaling_template.sh`
  - Minimal template for custom LUMI-G runs.

## Puhti Scripts

Existing `gputest_*` and long-run `train_*` scripts remain valid for Puhti workflows.

## Preflight Checks

Before submitting bridge/backbone runs:

- `python scripts/bridge_preflight_check.py --config <config.yaml>`
- `python scripts/report_bridge_errors.py --logs-dir logs/slurm --hours 72 --min-severity ERROR`
