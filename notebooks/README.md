# Notebooks Guide (Puhti/Mahti)

Lightweight workflow for shape checks, quick experiments, and debugging without long Slurm queues.

## Quick start (CPU)
- One-time: `PYTHONPATH=models python -m ipykernel install --user --name pf_models`.
- Launch: `PYTHONPATH=models ipython` or `jupyter notebook --no-browser --port 8888 --ip 0.0.0.0`.
- Use the smoke dataset/configs (`configs/data/phase_field_data_smoke.yaml`, `configs/train/train_smoke.yaml`) for fast shape checks.

## Interactive GPU shell (recommended for notebook sessions)
- Reserve a short GPU slot: `srun --account=project_2008261 --partition=gputest --gres=gpu:v100:1 --cpus-per-task=6 --time=00:30:00 --pty bash`.
- Activate env + path: `source physics_ml/bin/activate` (or use the full interpreter path), then `export PYTHONPATH=models`.
- Start Jupyter inside the allocation: `PORT=8888 jupyter lab --ip=0.0.0.0 --no-browser --port=$PORT`.
- Tunnel from laptop: `ssh -L 8888:$(hostname):8888 <user>@puhti.csc.fi` (change host for Mahti). Open the forwarded URL in your browser.
- For multi-hour tinkering, swap `--partition=gputest` for `--partition=gpu` and adjust `--time`.

## Notebook hygiene
- Keep scratch notebooks under `notebooks/` and clear large outputs before committing.
- For quick shape probes, prefer small, single-purpose notebooks (load one batch, inspect tensors, plot a couple slices).
- Save longer notes to a dated log (e.g., `notebooks/log_YYYYMMDD.md`) to track ideas/decisions without hunting through `.ipynb`.
- If you need custom deps, install them into `physics_ml/` with `pip install --user --target` or a venv clone to avoid polluting the shared env.

## Useful snippets
- Build a loader quickly:
  ```python
  import yaml
  from models.train.core.pf_dataloader import build_dataloaders

  cfg = yaml.safe_load(open("configs/train/train_smoke.yaml"))
  train_loader, val_loader, _ = build_dataloaders(cfg["data"])
  batch = next(iter(train_loader))
  print({k: v.shape for k, v in batch.items()})
  ```
- Instantiate a backbone for shape checks:
  ```python
  from models.backbones.registry import build_backbone
  from models.train.core.config import DotDict

  model = build_backbone(DotDict(cfg["model"]))
  out = model(batch["x"], t=None, cond=batch.get("cond"))
  print(out.shape)
  ```
- Run a one-step training smoke: `PYTHONPATH=models python -m models.train.core.smoketest_train -c configs/train/train_smoke.yaml` (CPU-friendly).

## When to Slurm vs. interactive
- Use the interactive GPU shell for <30–60 min explorations, plotting, and shape debugging.
- Use `sbatch slurm/train_smoke.sh` for regressions or if you need to leave a job unattended.
- For multi-node or long runs, keep using the existing Slurm launchers; capture notes/results in `docs/EXPERIMENT_STATUS.md` and a dated log in `notebooks/`.

## Keeping track
- Maintain a simple checklist in `notebooks/todo.md` (planned) and jot session notes per day.
- After a notebook proves an idea, port minimal changes into the Python modules/configs and add/adjust tests in `tests/` to avoid regressions.
