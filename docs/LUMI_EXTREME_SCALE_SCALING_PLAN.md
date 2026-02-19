# LUMI Extreme-Scale Scaling Plan

## Goal
Establish reproducible scaling evidence on LUMI-G (AMD MI250X) for phase-field surrogate training.

## Which workloads to benchmark
1. Autoencoder (AE) training:
- Purpose: fast sanity checks for ROCm portability and data pipeline throughput.
- Scope: single-node and light multi-node.

2. Latent-space diffusion/flow backbone training:
- Purpose: primary scaling evidence for Extreme Scale readiness.
- Scope: full strong/weak scaling campaign.

3. End-to-end latent workflow (AE encoder/decoder + backbone):
- Purpose: one integrated proof point that the full pipeline works at scale.
- Scope: limited runs (not full grid).

## Recommended benchmark matrix
1. Single-node smoke:
- GPUs: 1, 2, 4, 8 (one LUMI-G node)
- Fixed config, short duration (10-20 minutes)
- Validate loss trends, throughput stability, and no ROCm runtime errors.

2. Strong scaling (main evidence):
- Keep global batch fixed.
- Nodes: 1, 2, 4, 8 (then higher if queue allows).
- Report: samples/s, sec/step, scaling efficiency.

3. Weak scaling:
- Keep per-GPU batch fixed.
- Increase nodes proportionally.
- Report: throughput growth and communication overhead trend.

## Metrics to record
- Wall-clock sec/step (median and p95)
- Samples/sec and effective tokens/voxels/sec (if applicable)
- GPU memory headroom
- Data loading time fraction
- Convergence proxy at fixed step budget
- Failure rate / restart rate

## Acceptance criteria (proposal-ready)
- Stable runs at >= 8 nodes for backbone training.
- Strong-scaling efficiency trend documented (not necessarily perfect).
- Weak-scaling trend documented with bottleneck notes.
- Reproducible commands and exact config hashes tracked.

## Operational notes
- Keep Puhti for rapid debug only; use LUMI runs for final scaling evidence.
- Keep one canonical config family for scaling to avoid apples-to-oranges comparisons.
- Do not modify training semantics between scaling points.

## Suggested execution order
1. LUMI single-node smoke for AE and backbone.
2. Backbone strong scaling to the largest reliable node count.
3. Backbone weak scaling.
4. One end-to-end integrated latent workflow run.
5. Summarize in a benchmark table for application material.
