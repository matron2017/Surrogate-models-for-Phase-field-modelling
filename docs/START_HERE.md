# Start Here

Follow this order in a fresh session.

## 1. Check State

- `git status --short --branch`
- `hostname`
- `pwd`

## 2. Read Core Context

- `README.md`
- `docs/REPO_MAP.md`
- `docs/GIT_WORKFLOW.md`

## 3. Verify Active Jobs (do not disturb running training)

- `squeue -u $USER`
- `ls -lt logs/slurm | head`

## 4. Run Small Validation Before Long Jobs

- CPU test (fast):
  - `PYTHONPATH=models python -m pytest tests/test_backbones_rs.py -q`
- Slurm smoke run:
  - submit a `gputest` script from `slurm/` first.

## 5. Submit Long Runs Only After Smoke Success

- Puhti long partition: `gpu`
- Keep config and launcher edits in the same commit.
- Monitor first minutes of stdout/stderr before leaving job unattended.

## 6. LUMI Path

When preparing LUMI runs, use:

- `docs/LUMI_BENCHMARK_PREP_2026.md`
- `docs/LUMI_EXTREME_SCALE_SCALING_PLAN.md`
- `slurm/lumi_g_*.sh`

## Common Confusions

- `slurm/` contains submit scripts; `logs/slurm/` contains job output.
- `results/`, `mlruns/`, `runs*` are generated artifacts, not source.
- Many YAML files still contain Puhti absolute paths; LUMI launchers patch them at runtime.
