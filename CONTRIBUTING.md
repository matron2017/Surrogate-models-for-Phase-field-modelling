# Contributing Guide

This repository is used for active HPC training and analysis. Keep changes small, reproducible, and easy to audit.

## 1) Scope and Branching

- Work from `main` unless a separate branch is explicitly needed.
- Keep one topic per commit (for example: one backbone fix, one doc cleanup, one new config family).
- Do not mix code refactors with result regeneration in the same commit.

## 2) Before You Push

- Run at least a fast CPU test:
  - `PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests/test_backbones_rs.py -q`
- For trainer/schedule/loss changes, run the relevant suite:
  - `PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3.11 -m pytest tests -q`
- For Slurm training changes, run a `gputest` smoke job before long `gpu` jobs.

## 3) Config + Slurm Hygiene

- Keep config changes in `configs/train/*.yaml`.
- Keep launcher changes in `slurm/*.sh`.
- Job outputs must go to `logs/slurm/` via `%x_%j.out` and `%x_%j.err`.
- Prefer new config files over editing historical baselines in-place when running ablations.

## 4) Artifacts and Large Files

Generated artifacts must not be committed:

- `runs*`, `results/`, `logs/slurm/*.out`, `logs/slurm/*.err`
- `mlruns/*` (except tracked placeholders/readmes)
- large data/model files (`*.h5`, `*.pth`, `*.pt`, `*.npy`)

Use committed markdown summaries and small JSON/CSV summaries for traceability instead.

## 5) Documentation Expectations

- If behavior changes, update at least one of:
  - `README.md`
  - `docs/README.md`
  - `docs/WORKFLOW_THERMAL_LATENT.md`
- For operational changes, also update:
  - `slurm/README.md`
  - `docs/DEV_GUIDE.md`

## 6) Recommended Commit Structure

1. Code/config changes
2. Tests
3. Docs

Write commit messages in imperative form, e.g.:

- `train: add UniDB predict-next no-mass config`
- `loss: harden denominator guards for vrmse and mass error`
- `docs: add git workflow and contribution guide`
