# LUMI Benchmark Prep (2026-02-19)

This note defines what to move from Puhti to LUMI first, what to benchmark, and
what evidence to collect for an Extreme Scale application.

## 1. Scope

Primary target for scaling evidence:
- Latent-space backbone training (flow-matching / diffusion).

Secondary target:
- AE smoke and light scaling for portability verification.

## 2. Minimal Transfer Bundle (recommended first)

Code (small):
- `configs/`, `models/`, `slurm/`, `scripts/`, `docs/`, `tests/`, `README.md`

Data/checkpoints (benchmark-critical):
- `data/latent_best_psgd_e279_dev/train_latent_experimental_midtrain.h5`
- `data/latent_best_psgd_e279_dev/val_latent_experimental_midtrain.h5`
- `data/latent_best_psgd_e279_dev/test_latent_experimental_midtrain.h5`
- `runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth`

Observed Puhti size:
- Latent HDF5 bundle: ~5.1G
- Canonical AE checkpoint: ~1.1G
- Total practical first transfer: ~6.2G (+ small metadata)

## 3. Do Not Transfer Initially

- `data/stochastic/` full tree (~90G)
- historical `logs/`, `results/`, `mlruns/`
- old checkpoints not needed for benchmark reproducibility

## 4. LUMI-G Constraints That Affect Scripts

LUMI-G compute node:
- 56 CPU cores
- 8 GPU GCDs (4x MI250X, 2 GCD per GPU)

For full-node GPU runs, use:
- `--ntasks-per-node=8`
- `--gpus-per-node=8`
- `--cpus-per-task=7`

For multi-GPU jobs, keep one rank per GCD and use:
- `--gpu-bind=closest`
- `--cpu-bind=cores`

## 5. Current Practical Benchmark Capacity (project_462001306)

From account/partition checks on 2026-02-19:
- `small-g`: max 4 nodes, max 3 days
- `standard-g`: max 1024 nodes, max 2 days
- `dev-g`: max 32 nodes, max 3 hours
- assoc limits include up to ~200 running jobs on `small-g`/`standard-g`

Realistic campaign now:
1. functional smoke on `dev-g` or `small-g`
2. 1/2/4 node scaling on `small-g`
3. 8+ node points on `standard-g`

## 6. Benchmark Evidence Required

Collect for each run point:
- wall-time per step (median and p95)
- throughput (samples/s)
- effective batch and accumulation steps
- node count, world size, and per-GPU batch
- success/failure status and restart count
- fixed git commit + exact YAML used

## 7. Recommended Execution Order

1. AE smoke (`slurm/lumi_g_ae_smoke.sh`)
2. Backbone smoke (`MODE=strong`, 1 node)
3. Backbone strong scaling: 1 -> 2 -> 4 -> 8 nodes
4. Backbone weak scaling: 1 -> 2 -> 4 -> 8 nodes
5. One integrated end-to-end run

## 8. Resource Estimation Formula

- `GPU-hours = nodes * gpus_per_node * walltime_hours * run_count`
- `CPU-core-hours = nodes * (ntasks_per_node * cpus_per_task) * walltime_hours * run_count`

Add buffer for queue delay and failed runs before submitting final request numbers.
