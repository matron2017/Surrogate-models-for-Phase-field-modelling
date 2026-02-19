# Git Workflow (Practical)

This project often has many simultaneous experiments. Use this workflow to avoid accidental commits and broken launch scripts.

## 1) Start of Session

From repo root:

```bash
git status --short --branch
```

Quickly classify changes into:

- tracked source/config/docs edits
- generated artifacts (should stay ignored)
- unrelated leftovers from old experiments

## 2) Stage by Topic

Use explicit paths; avoid `git add .` in this repository.

Examples:

```bash
git add models/train/core/train.py models/train/loss_registry.py
git add configs/train/train_diffusion_bridge_unet_thermal_*.yaml
git add docs/README.md docs/WORKFLOW_THERMAL_LATENT.md
```

Then verify:

```bash
git diff --staged --name-only
```

## 3) Validate Before Commit

Minimum:

```bash
PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests/test_backbones_rs.py -q
```

If touching trainer/schedule/loss:

```bash
PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests -q
```

If touching Slurm/config logic:

- run a `gputest` smoke launch first
- only then submit long `gpu` jobs

## 4) Commit

Keep one concern per commit:

- code correctness
- experiment config
- docs/workflow

Preferred message format:

- `area: short imperative summary`

Examples:

- `diffusion: enforce UniDB t=1..T-1 sampler policy logging`
- `metrics: add endpoint metric logging flags`
- `docs: clarify slurm launch vs logs/slurm outputs`

## 5) Final Pre-Push Check

```bash
git status --short
```

Make sure nothing unintended is staged (especially generated plots/logs/results).

## 6) HPC-Specific Note

Submitted Slurm jobs read config/code at runtime from filesystem. If a job is pending and you change relevant files, the runtime behavior may change without re-submission. Re-submit intentionally when you need an immutable launch record.
