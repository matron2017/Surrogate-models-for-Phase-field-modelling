# LUMI vs Puhti Technical Cheat Sheet (2026-02-19)

Use this when switching scripts/configs between Puhti and LUMI.

## 1. Node and GPU layout

Puhti GPU node (observed from `scontrol show node`):
- `CPUTot=40`
- `Gres=gpu:v100:4`
- typical script pattern: `--gres=gpu:v100:<n>`

LUMI-G node (observed from `scontrol show node`):
- `Gres=gpu:mi250:8` (8 GPU units per node)
- scheduler-visible CPU total `128`; LUMI job recommendations for full GPU use are
  typically `8 tasks` x `7 CPUs` per task for balanced placement.
- script pattern: `--ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7`

## 2. Key partition differences

Puhti:
- `gputest`: max 2 nodes, 15 min
- `gpu`: max 20 nodes, 3 days

LUMI:
- `dev-g`: max 32 nodes, 3 hours
- `small-g`: max 4 nodes, 3 days
- `standard-g`: max 1024 nodes, 2 days

## 3. Runtime stack differences

Puhti workflow in this project:
- CUDA/V100-oriented scripts and env variables
- many launchers pinned to `/scratch/project_2008261/...` and `project_2008261`

LUMI workflow (recommended):
- ROCm stack via modules:
  - `module load LUMI partition/G Local-CSC/default pytorch/2.5`
- optional project venv overlay:
  - `.venv_physics_ml_lumi`
  - bootstrap: `scripts/lumi_setup_physics_ml_env.sh`

## 4. Launcher differences to remember

Do not reuse Puhti launcher files directly on LUMI.

Use LUMI launchers:
- `slurm/lumi_g_ae_smoke.sh`
- `slurm/lumi_g_backbone_scaling_job.sh`
- `slurm/lumi_g_submit_backbone_scaling_matrix.sh`

They support:
- path patching from Puhti-root to LUMI-root
- strong/weak scaling modes
- optional auto-activation of `.venv_physics_ml_lumi`

## 5. Diagnostics toolbox on LUMI

- Scheduler/accounting: `squeue`, `sacct`, `sstat`
- GPU/profiling tools available: `rocprof`, `rocprofv2`, `rocm-smi`
- Cray profiling modules: `perftools-lite*`
- Project helpers:
  - `scripts/lumi_collect_scaling_metrics.sh`
  - `scripts/lumi_quota_snapshot.sh`

## 6. Data sanity for current latent workflow

Canonical latent dataset provenance:
- source pixel HDF5: `data/stochastic/rightclean/simulation_{train,val,test}_rightclean_fixed34_gradshared.h5`
- latent HDF5 output: `data/latent_best_psgd_e279_dev/*_latent_experimental_midtrain.h5`
- AE checkpoint: `runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth`

