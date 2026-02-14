# Repository Operation Instructions

- Start every session at the repo root `/scratch/project_2008261/pf_surrogate_modelling`.
- Bootstrap context first: run `tools/codex_dir_check.sh --depth 2` (uses `/scratch/project_2008261/physics_ml/bin/python3` to print the tree + doc headers). If python is missing, surface that instead of guessing paths.
- Use the Puhti/Mahti env at `/scratch/project_2008261/physics_ml/bin/python3` (avoid the system Python 3.6). Set `PYTHONPATH=models` for code/tests.
- Prefer `rg`/`ls` over guessing. Consult `README.md` plus `docs/DIRECTORY_GUIDE.md`, `docs/DEV_GUIDE.md`, `docs/ENVIRONMENT.md`, `docs/EXPERIMENT_STATUS.md`, and `docs/NEXT_STEPS.md` before making changes.
- Tests: `PYTHONPATH=models /scratch/project_2008261/physics_ml/bin/python3 -m pytest -q tests` (CPU). For cluster runs, use one of the active AE launchers in `slurm/`.
- Slurm helpers live in `slurm/`; data/runs are symlinked to `/scratch/project_2008261/solidification_modelling/...` per `README.md`.
- Optional shell helper: add `alias pf_boot='cd /scratch/project_2008261/pf_surrogate_modelling && tools/codex_dir_check.sh --depth 2'` to your shell profile to refresh context on login.

## Prompting/Workflow References

- Prompting best-practice pattern: clear task scope, verification steps, task splitting, and debugging workflow.
- Repository task framing guide: concise engineering requests with reproducible command outcomes.
- AGENTS.md guide — global + per-repo instructions for persistent context.
- CLI usage, flags, config, approval policy defaults, and reproducible command execution patterns.
- General prompt engineering best practices — succinct instructions, examples, and verification steps.
