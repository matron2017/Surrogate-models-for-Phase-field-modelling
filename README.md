# Surrogate Models for Phase-Field Modelling

## Rapid Solidification Modelling Workspace

This directory now mirrors only the rapid solidification workflow, with the Python package renamed to `models/` (formerly `pf_surrogate_modelling`). For detailed guides and checklists, see `docs/README.md`. Key layout:

- `models/` – Python package housing backbones/conditioning, diffusion + flow-matching operators, trainer, datapipes helpers, tools/visuals.
- `configs/` – YAML for the phase-field dataset, model training (`configs/train/*.yaml`), and visuals.
- `slurm/` – ready-to-run Slurm launchers that point at `models/train/core/train.py`.
- `slurm/benchmark_flowmatch_uafno_scaling.sh` – gputest scaling smoke for the ~202M UAFNO (1/2/4/8 GPUs torchrun sweep).
- `visuals/basic/` – lightweight evaluation and inspection tools (with diagnostics subfolder).
- `visuals/hq/` – high-quality plotting utilities.
- `experiments/diffusion_prototype/` – Hugging Face diffusers sandbox (placeholder data, configs, DDPM scripts).
- `docs/` – consolidated developer, environment, monitoring, and checklist docs.
- `docs/PUHTI_PARTITIONS.md` – cheat sheet for Puhti partitions (gputest/gpu/gpusmall) and GPU request syntax.

Heavy artefacts are re-used via symlinks:

- `data/rapid_solidification -> ../solidification_modelling/data/rapid_solidification`
- `runs -> ../solidification_modelling/runs`
- `runs_debug -> ../solidification_modelling/runs_debug`
- `physics_ml -> ../solidification_modelling/physics_ml`

All training and evaluation launchers now default to the code in this directory. Update the configs or symlinks if you change the workspace root again.

## Directory Map (Quick Reference)

| Path | Purpose |
|------|---------|
| `configs/data/phase_field_data*.yaml` | Phase-field dataset descriptors copied from the Mahti builders; keep Puhti paths in sync with `/scratch/project_2008261/rapid_solidification/data`. |
| `configs/train/*.yaml` | Experiment descriptors (model family, backbone, diffusion task settings, loss weights). Smoke configs (`*_smoke.yaml`) stay ≤15 min on `gputest`. |
| `datapipes/` | Slurm/CLI wrappers calling `models/datapipes/*.py` for VTU→HDF5 conversion and wavelet weights. |
| `docs/` | Centralised developer guides, environment notes, monitoring cheatsheets, and active checklists. |
| `models/backbones/` | Backbone implementations referenced by the registry (`UNet`, `UAFNO`, placeholders for SSM/SineNet). |
| `models/conditioning/` | Conditioning layers shared across the backbones. |
| `models/diffusion/` | Noise schedules and timestep samplers plus pluggable operators/configs for rollout tasks. |
| `models/train/core/train.py` | Canonical trainer/orchestrator with MLflow logging, descriptor plumbing, and resume-aware parent-run IDs. |
| `models/train/tasks/` | Task wrappers (e.g., diffusion rollouts) around registry-built backbones. |
| `slurm/` | Puhti-ready SLURM scripts. Test and smoke scripts default to `gputest` with `--gres=gpu:v100:1`; edit only when queueing long `gpu` jobs. |
| `slurm/benchmark_flowmatch_uafno_scaling.sh` | Torchrun scaling sweep (1/2/4/8 GPUs) for the ~202M UAFNO depth-8 config on gputest (15 min cap). |
| `visuals/basic/run_dataset_eval_and_plots.py` | Standard evaluation entrypoint logging `metrics.json`/plots for a checkpoint + config pair. |
| `tests/` | CPU-only contract tests (`pytest -q`) for dataloaders, backbones, diffusion schedules/samplers, and the tiny training loop. |
| `experiments/diffusion_prototype/` | Residual diffusion playground (VAE + DDPM scripts, configs, placeholder data). |

## Quick Start (Puhti)

1. **Prep smoke data** – `sbatch datapipes/smoke_build.sh` (writes the tiny VTU→HDF5 bundle defined in `configs/data/phase_field_data_smoke.yaml`).
2. **Run smoke training** – `python -m models.train.core.train -c configs/train/train_smoke.yaml` or `sbatch slurm/train_smoke.sh` (≤15 min on `gputest`, `gpu:v100:1`, `cpus-per-task=8`).
3. **Visual sanity check** – `sbatch visuals/basic/run_smoke.sh` to plot metrics for the smoke checkpoint.
4. **CPU tests** – `pytest -q tests` before large refactors; these run without GPUs.
5. **Diffusion smoke (residual DDPM)** – `sbatch slurm/train_diffusion.sh` (runs `experiments/diffusion_prototype/src/train_ddpm_residual.py` on the placeholder HDF5, GPU-friendly via `gputest`).
6. **Diffusion inference** – `python experiments/diffusion_prototype/src/infer_ddpm_residual.py --config ... --checkpoint runs_debug/diffusion_smoke/epoch001.pt` to run the DDPM denoising loop and report patch-level MSE.

Use `run_next_steps.sh` to enqueue datapipes/train/visuals smoke jobs back-to-back.

**Puhti data note** – The Mahti VTU/HDF5 bundle has not been copied to `/scratch/project_2008261/alloy_solidification/data/pf_data` yet, so datapipes jobs should continue using the placeholder HDF5 dataset already checked in under `data/rapid_solidification`. Once the real VTU data is available, create the missing directory on Puhti, re-run `datapipes/smoke_build.sh`, and then re-try the chained smoke script.

**Slurm logs** – All submission scripts write stdout/stderr to `logs/slurm/%x_%j.{out,err}` so the workspace root stays tidy. Remove or archive old `*.out` files after migrating.

### Interactive Pipeline Launch

Run the guarded config editor from the repository root so relative `sbatch` paths resolve:

```bash
cd /scratch/project_2008261/ML_workflow
PYTHONPATH=src env/physics_ml/bin/python3.11 src/project_time_dependent/pipeline.py
```

### Multi-Node Parallel Test

- `train/core/ddp_multi_node_check.py` prints `(rank, hostname, local_rank)` for every process and performs an `all_reduce`. It verifies NCCL connectivity across multiple nodes before launching heavier training.
- `slurm/test_parallel_multinode.sh` submits a Puhti `gputest` job (2 nodes × 4 V100 GPUs, 15‑minute limit), runs the DDP check, and then executes `train/core/parallel_solid_data.py` across all 8 GPUs using the placeholder HDF5 dataset before emitting `seff`/`sacct` diagnostics.
- Submit via `sbatch slurm/test_parallel_multinode.sh`, then `squeue -j <jobid>` to monitor. The output file `logs/slurm/%x_%j.out` captures both steps so you can confirm hostnames, ranks, and throughput.
- If the DDP check logs only one hostname (single node), re-queue on the `gpu` partition and ensure the project has quota for multi-node GPU jobs. For production-length runs, change the partition/time back to `gpu` since `gputest` enforces a strict 15‑minute wall clock.
- For GPU utilisation tooling (`seff`, `sacct`, live `nvidia-smi`, Nsight profiling, scaling tests), refer to `docs/GPU_MONITORING.md`.

## Cluster Session Checklist (Codex)

Run this quick checklist whenever you start a fresh Codex session so Puhti/Mahti submissions behave the same way as the manual shell:

1. **Confirm the host** – `hostname` should read `mahti-loginXX` or `puhti-loginXX`. Anything else means you are in a sandbox without Slurm access.
2. **Verify Slurm CLI** – `which sbatch` and `sbatch --version`; both clusters expose `/usr/bin/sbatch` (Mahti 24.05.7, Puhti 24.05.8).
3. **List available accounts** – `sacctmgr list user $USER` and `sacctmgr show associations where user=$USER format=cluster,user,account%-30,partition`. Note the project string that belongs to the cluster you are on (e.g. `project_2008261` works on both Mahti and Puhti, while Puhti also offers `project_2003809`/`project_2003810`).
4. **Filesystem sanity** – `pwd` should sit under `/scratch/project_2008261/...`; `ls` should show this directory tree so symlinks resolve correctly.
5. **Smoke submit** – launch the tiny Slurm wrap to prove the account/partition pair is valid:

   ```bash
   cd /scratch/project_2008261/rapid_solidification
   sbatch --account=project_2008261 --partition=gputest \
          --gres=gpu:v100:1 --cpus-per-task=4 --time=00:01:00 \
          --wrap="hostname"
   ```

   Swap `project_2008261` for any other account from step 3 when needed. The job should exit within a few seconds and write `slurm-<jobid>.out` with the compute hostname.

### Latest Verified Outputs

| Cluster | Host | Slurm | Account | Test Job |
|---------|------|-------|---------|----------|
| Mahti | `mahti-login11.mahti.csc.fi` | `/usr/bin/sbatch` (24.05.7) | default `project_2008261` | `sbatch --account=project_2008261 ...` → Job `5547125` (gputest, `gpu:a100:1`). |
| Puhti | `puhti-login11.bullx` | `/usr/bin/sbatch` (24.05.8) | `project_2008261` (also `project_2003809`, `project_2003810`) | `sbatch --account=project_2008261 ...` → Job `30877443` (gputest, `gpu:v100:1`) writing `slurm-30877443.out`. |

If any step fails, re-launch Codex directly on the cluster login node with your user and re-run the checklist before attempting the longer scripts in `slurm/`.

## Slurm Job Watcher

Monitor Puhti/Mahti jobs from your laptop without polling manually:

```bash
python tools/watch_slurm_job.py puhti 30869387
python tools/watch_slurm_job.py mahti 5547125 --interval 120
```

The helper uses `squeue` for live updates, falls back to a single `sacct` query
when the job leaves the queue, and rings the terminal bell once the final state
is known.

## Diffusion Config Notes

- `diffusion.noise_schedule` accepts `linear`, `cosine`, `logsnr_laplace`, or `learned`; each maps to the factory in `models/diffusion/scheduler_registry.py`.
- `diffusion.timestep_sampler` looks up samplers via `models/diffusion/timestep_sampler.py`; schedule and sampler kwargs live under `diffusion.schedule_kwargs`/`diffusion.sampler_kwargs`.
- Adaptive hooks come from `models/train/adaptive/registry.py` using `adaptive.region_selector` and optional `adaptive.region_kwargs`.
- **Trainer wiring** – `models/train/core/train.py` reads these fields once, builds the schedule/sampler/region selector via the registries, and logs the descriptors to MLflow (when enabled). Set `task.name: diffusion` to wrap any backbone in the pluggable diffusion rollout (dt/n_steps/thermal_bc).

## Legacy Scripts

Legacy helpers now live under `/scratch/project_2008261/solidification_modelling/scripts_legacy/` (snapshot taken before the rapid_solidification refactor). Treat them as read-only references—every new run (datapipes, training, visuals, tests) should go through the `rapid_solidification/` entrypoints documented above.

## GitHub Publishing Plan

Upstream now lives at `git@github.com:matron2017/Surrogate-models-for-Phase-field-modelling.git` (public). To stay aligned with CSC/Puhti and avoid path drift:

1. **Set the remote** – `git remote add origin git@github.com:matron2017/Surrogate-models-for-Phase-field-modelling.git` (or `git remote set-url origin ...` if it already exists). Use the Puhti SSH key at `/scratch/project_2008261/ssh_keys/Puhti_surrogate_key` (fingerprint `SHA256:D2KnQYxnDQX0Rgsaozu1/gg6A00MSdQdTdnAG9WyJcY`).
2. **Canonical root** – all configs/scripts use `/scratch/project_2008261/pf_surrogate_modelling`; avoid reintroducing `/scratch/project_2008261/models` paths now that the symlink was removed.
3. **Push** – `git push origin main` after staging only the intended files (configs, slurm scripts, docs); heavy artefacts remain excluded via `.gitignore`.
4. **Tests before pushes** – run `PYTHONPATH=models python -m pytest tests/test_backbones_rs.py -q` (fast) or `tests -q` (full CPU), and on GPU boxes submit `slurm/train_smoke.sh` for a quick sanity.
5. **Future CI** – mirror the smoke test (`pytest -q tests`) and debug train command in GitHub Actions once secrets are set up, using the Tykkÿ container recipe in `docs/ENVIRONMENT.md` for reproducibility.

This keeps the workflow transparent and lets contributors fork/PR without touching CSC-internal paths.
