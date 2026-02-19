# LUMI Benchmark Prep (2026)

This note answers: what to move from this repo to LUMI firstwhat not to move, and what benchmark evidence to collect for CSC applications.

## 1) Call/Access Mode Reality Check (as of 2026-02-17)

Your pasted text is from the first 2025 call.  
Current public info indicates a newer Finnish Extreme Scale call for 2026 with updated limits/deadline.

- LUMI news (Finnish Extreme Scale call 2026deadline extension):  
  https://lumi-supercomputer.eu/second-finnish-lumi-extreme-scale-call-opens/
- LUMI access modes for Finnish users (Regular/Benchmark/Extreme):  
  https://www.lumi-supercomputer.eu/users-in-finland/
- CSC resource calls/access guidance:  
  https://research.csc.fi/lumi-access

Before drafting the final applicationre-check these pages the same day you submit.

## 2) What This Repo Actually Needs For Benchmarking

### Minimal transferable benchmark bundle (recommended first pass)

- Code/config/scripts/docs/tests:
  - `configs/``models/`, `slurm/`, `scripts/`, `docs/`, `tests/`, `README.md`
- Canonical latent dataset family:
  - `data/latent_best_psgd_e279_dev/train_latent_experimental_midtrain.h5` (~3.5G)
  - `data/latent_best_psgd_e279_dev/val_latent_experimental_midtrain.h5` (~882M)
  - `data/latent_best_psgd_e279_dev/test_latent_experimental_midtrain.h5` (~846M)
- Canonical AE checkpoint paired with this latent dataset:
  - `runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth` (~1.1G)

Observed size summary in this workspace:

- `data/latent_best_psgd_e279_dev`: ~5.1G
- Canonical AE checkpoint: ~1.1G
- Code+configs+scripts+docs+tests: small (few MB)

Total practical first transfer: ~6.2G + metadata.

### Optional (only if needed)

- Warm-start training checkpoints (`~4.8G` each) from smoke/previous runs.

## 3) What Not To Move Initially

- `data/stochastic` (~90G) unless you need raw/full-space re-encoding workflows on LUMI immediately.
- `logs/slurm/``results/`, `mlruns/` historical artifacts.
- `runs` symlink targets in bulk.

Move only the exact checkpoints/datasets required for benchmark reproducibility.

## 4) Porting Gaps You Must Fix Before LUMI Benchmarks

Current repo has many Puhti-specific and CUDA/V100 assumptions.

- Hardcoded `/scratch/project_2008261` occurrences in code/config/scripts: `366`
- Many Slurm launchers hardcode:
  - `#SBATCH --gres=gpu:v100:...`
  - CUDA/NVIDIA-specific env (`CUBLAS_WORKSPACE_CONFIG``PYTORCH_CUDA_ALLOC_CONF`, etc.)
- Python env here is CUDA wheels (`torch==2.8.0+cu128`)not ROCm.

For LUMI-G you should prepare:

1. Cluster-agnostic path variables (`PROJECT_ROOT``DATA_ROOT`, `RUNS_ROOT`).
2. LUMI Slurm templates (LUMI partition/account syntax from LUMI docs).
3. ROCm-compatible PyTorch environment/container.
4. One canonical benchmark launcher per objective (ReFlow-style and SFM-style).

## 5) Benchmark Evidence To Collect (for CSC technical readiness)

Use Benchmark Access first to produce these artifacts:

1. **Single-node baseline**:
   - throughput (samples/s or steps/s)
   - GPU memory usage
   - convergence sanity for at least a few epochs
2. **Strong scaling**:
   - fixed global batchnodes: 1 -> 2 -> 4
   - report speedup and parallel efficiency
3. **Weak scaling**:
   - fixed per-GPU batchnodes: 1 -> 2 -> 4
   - report throughput growth and stability
4. **I/O profile**:
   - startup + dataloader overhead
   - sustained read behavior from dataset location
5. **Reproducibility package**:
   - exact commit hash
   - exact config YAMLs
   - exact Slurm scripts
   - exact env export

## 6) Resource Estimation Template

Use measured benchmark wall times (not guesses) to extrapolate:

- `GPU-hours = (#nodes) * (GPUs per node) * (walltime hours) * (number of runs)`
- `CPU-core-hours = (#nodes) * (cores per node used) * (walltime hours) * (number of runs)`
- `Storage-hours = (requested TiB) * (project duration hours)`

Include queue-time risk and failed-run overhead buffer.

## 7) Practical Transfer Checklist

1. Export code snapshot at fixed commit.
2. Export env lockfile (`pip freeze` / container recipe).
3. Copy minimal dataset + canonical AE checkpoint.
4. Validate one tiny smoke run on LUMI.
5. Run the scaling benchmark matrix.
6. Archive plots/tables/log extracts for proposal appendix.

## 8) Immediate Recommendation For This Project

Given current workflowstart with:

- One ReFlow-style benchmark run config:
  - `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_reflow2506_long_v1.yaml`
- One SFM-style benchmark run config:
  - `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_sfm_latent_long_v1.yaml`

and benchmark both with the same data split and reporting format.

