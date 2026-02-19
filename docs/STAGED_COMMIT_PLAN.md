# Staged Commit Plan (Current Worktree)

Use this plan to avoid mixing unrelated changes in one commit.

## 0) Inspect

```bash
git status --short --branch
```

## 1) Meta + Workflow Docs (safe first commit)

Files:

- `.gitignore`
- `.editorconfig`
- `.gitattributes`
- `CONTRIBUTING.md`
- `README.md`
- `docs/README.md`
- `docs/START_HERE.md`
- `docs/GIT_WORKFLOW.md`
- `docs/STAGED_COMMIT_PLAN.md`
- `slurm/README.md`

Commands:

```bash
git add .gitignore .editorconfig .gitattributes CONTRIBUTING.md
git add README.md docs/README.md docs/START_HERE.md docs/GIT_WORKFLOW.md docs/STAGED_COMMIT_PLAN.md
git add slurm/README.md
git diff --staged --name-only
```

Suggested message:

`docs: add repo workflow meta files and clean gitignore`

## 2) Numerical Safety (loss/scheduler/metrics + tests)

Files:

- `models/diffusion/scheduler_registry.py`
- `models/train/loss_registry.py`
- `models/train/core/loss_functions.py`
- `models/train/core/metric_stats.py`
- `tests/test_numerical_safety_rs.py`

Commands:

```bash
git add models/diffusion/scheduler_registry.py
git add models/train/loss_registry.py models/train/core/loss_functions.py models/train/core/metric_stats.py
git add tests/test_numerical_safety_rs.py
git diff --staged --name-only
```

Suggested message:

`train: harden denominator guards and add numerical safety tests`

## 3) UniDB Training/Logging Update (if committing now)

Files (from current active changes):

- `models/train/core/train.py`
- `configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu7h_ddp8_b80_controlhint_25steps.yaml`
- `configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu7h_ddp8_b80_controlhint_predictnext_40steps.yaml`
- `configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b80_rdbmres_predictnext_nomass.yaml`

Commands:

```bash
git add models/train/core/train.py
git add configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu7h_ddp8_b80_controlhint_25steps.yaml
git add configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu7h_ddp8_b80_controlhint_predictnext_40steps.yaml
git add configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b80_rdbmres_predictnext_nomass.yaml
git diff --staged --name-only
```

Suggested message:

`diffusion: enable endpoint metrics and add predict-next no-mass UniDB config`

## 4) Validate Before Push

```bash
PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests/test_numerical_safety_rs.py tests/test_diffusion_components_rs.py tests/test_diffusion_loss_weighting_rs.py tests/test_flow_objectives_rs.py -q
```

For large training/script changes, run a `gputest` smoke job before long jobs.
