# LUMI Extreme-Scale Scaling Plan

Updated: 2026-02-19

## Objective

Produce reproducible strong/weak scaling evidence on LUMI-G for the latent
backbone training workflow, with AE results used as portability checks.

## External Requirements (summary)

For Extreme Scale applications, benchmark evidence is expected and benchmark
access is the standard route before requesting very large production resources.

LUMI-G job-layout requirements that matter for this project:
- 8 GPU GCDs per node (4x MI250X)
- 56 CPU cores per node
- preferred full-node layout: `8 tasks/node`, `1 GPU task`, `7 CPU cores/task`

## Workloads and Priority

1. Backbone scaling (primary evidence):
- latent flow-matching or diffusion UNet/UViT backbone

2. AE smoke/light scaling (secondary evidence):
- verify ROCm portability and data throughput

3. End-to-end integrated run:
- one proof run after scaling matrix completion

## Matrix Design

### A. Smoke

- Nodes: 1
- World size points: 1, 2, 4, 8 GCD
- Duration: 10-20 minutes
- Purpose: launch correctness, ROCm runtime stability, dataloader sanity

### B. Strong Scaling (main)

- Keep global effective batch fixed (e.g. 64 or 80)
- Nodes: 1 -> 2 -> 4 -> 8 (extend if queue allows)
- Report:
  - throughput speedup vs 1 node
  - parallel efficiency = speedup / node_count

### C. Weak Scaling

- Keep per-GPU batch fixed
- Increase node count proportionally
- Report:
  - throughput growth
  - sec/step drift
  - communication overhead trend

## Metrics to Record Per Run

- commit hash
- config hash/path
- partition/account/nodes/tasks
- per-GPU batch and accumulation
- sec/step (median, p95)
- samples/s
- memory headroom (from logs if available)
- run status (success/failure)

## Scripts in This Repository

- `slurm/lumi_g_ae_smoke.sh`
- `slurm/lumi_g_backbone_scaling_job.sh`
- `slurm/lumi_g_submit_backbone_scaling_matrix.sh`
- `slurm/lumi_g_scaling_template.sh`

## Recommended Run Order

1. AE smoke (`RUN_TASKS=1` then `2/4/8`)
2. Backbone strong-scaling matrix
3. Backbone weak-scaling matrix
4. One integrated latent workflow run
5. Export final table for proposal appendix

## Success Criteria for Proposal Material

- Stable multi-node backbone runs at >= 8 nodes
- Documented strong-scaling efficiency trend
- Documented weak-scaling trend with bottleneck interpretation
- Fully reproducible commands + configs + commit IDs
