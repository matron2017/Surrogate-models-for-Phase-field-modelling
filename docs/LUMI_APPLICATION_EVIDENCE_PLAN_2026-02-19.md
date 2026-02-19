# LUMI Extreme Scale Application Evidence Plan (2026-02-19)

This document maps the application template questions in
`/scratch/project_462001306/LUMI Extreme Scale application 2026 , PF-surrogate etc.docx`
to concrete benchmark tests, diagnostics, and deliverables for this project.

Project context:
- Project number: `462001306`
- Unix group: `project_462001306`
- Resources currently shown: `5000 GPUh`, `500000 CPU core-hours`, `22000 TiBh`
- Scientific target: latent surrogate training for PF rapid solidification (AE + latent dynamics)

## 1. What to Demonstrate for Extreme Scale

For technical sections in the application (`3.2.4.*`, `3.2.5.*`, `3.2.6.*`, `3.8`, `3.9`),
prepare evidence in five groups:

1. Portability and software readiness on LUMI-G (AMD MI250X)
2. Parallel scalability (strong + weak scaling)
3. Stability and reproducibility (repeatable runs, checkpoint/restart)
4. I/O behavior and storage footprint
5. Resource model and production plan linked to measured benchmark results

## 2. Question-to-Evidence Mapping

## 3.2.4.1 Software

Provide:
- Code stack: `models/`, `configs/`, `slurm/` in this repository
- Active entrypoint: `python -m models.train.core.train -c <config.yaml>`
- Workloads: AE latent compression + latent flow/diffusion backbone

Evidence:
- Successful LUMI-G smoke runs and multi-node runs with commit ID and configs
- Git commit references and exact launcher scripts

## 3.2.4.2 Dependencies / build requirements

Provide:
- LUMI modules used (`LUMI`, `partition/G`, `Local-CSC/default`, `pytorch/2.5`)
- Python package layer used on top (if any)

Evidence:
- `scripts/lumi_setup_physics_ml_env.sh`
- `docs/LUMI_ENV_MIGRATION_FROM_PUHTI_PHYSICS_ML.md`

## 3.2.4.3 Parallel and GPU programming

Provide:
- PyTorch DDP with one rank per GCD
- Slurm layout on LUMI-G: `--ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7`

Evidence:
- Strong scaling matrix and weak scaling matrix results
- Job scripts under `slurm/lumi_g_*.sh`

## 3.2.4.4 I/O requirements

Provide:
- HDF5 input size and read pattern
- Checkpoint frequency and per-checkpoint size
- Number of output files and expected growth

Evidence:
- HDF5 bundle size (`~5.1G`) and AE checkpoint size (`~1.1G`)
- Slurm diagnostics logs and output directory summaries

## 3.2.5.* Data management and workflow

Provide:
- Data flow: transfer -> training -> metrics extraction -> archiving
- Retention and post-project transfer plan (essential outputs only)
- Analysis tools and timing in the project workflow

Evidence:
- `scripts/lumi_collect_scaling_metrics.sh`
- Run manifest (job IDs, commit, config, metrics CSV/JSON)

## 3.2.6.1 Application performance

Provide:
- Benchmark system details (LUMI-G partitions, node counts)
- Measured throughput and sec/step trends
- Parallel efficiency and bottleneck interpretation

Evidence:
- Strong scaling speedup/efficiency table
- Weak scaling throughput table
- `sacct`/`sstat` based diagnostics

## 3.8, 3.9 Resource request / job model

Provide:
- Request rationale derived from measured benchmark runs
- Concurrent jobs, walltime, checkpoint cadence
- Storage and file count estimates

Evidence:
- Budget table in Section 5 below
- Final benchmark report table with measured run durations

## 3. Recommended Test Campaign

## A. Functional smoke (required first)

1. AE smoke (single node, `RUN_TASKS=1,2,4,8`)
2. Backbone smoke (single node, `RUN_TASKS=1,2,4,8`)
3. Basic checkpoint/restart test (kill/requeue from checkpoint)

Scripts:
- `slurm/lumi_g_ae_smoke.sh`
- `slurm/lumi_g_backbone_scaling_job.sh`

## B. Strong scaling matrix (primary evidence)

- Fixed global effective batch (example: 80)
- Nodes: `1, 2, 4, 8, 16` (optionally 32)
- Measure: sec/step, samples/s, efficiency

## C. Weak scaling matrix

- Fixed per-GPU batch (example: 1)
- Nodes: `1, 2, 4, 8, 16`
- Measure: throughput growth and sec/step drift

## D. Production-like pilot

- 1 to 3 longer runs on higher node counts
- Purpose: stability and realistic throughput for final resource request

## 4. Diagnostics Tools to Use

Scheduler and accounting:
- `squeue -u $USER`
- `sacct -j <jobid> --format=JobID,JobName,Partition,AllocTRES,ElapsedRaw,TotalCPU,MaxRSS,State,ExitCode`
- `sstat -j <jobid>.batch --format=AveCPU,AveRSS,MaxRSS,MaxVMSize`

GPU/runtime profiling on LUMI:
- `rocprof`, `rocprofv2`
- Cray perftools-lite modules: `perftools-lite`, `perftools-lite-gpu`, `perftools-lite-hbm`

Project helper scripts in this repo:
- `scripts/lumi_collect_scaling_metrics.sh`
- `tools/watch_slurm_job.py` (queue/watch helper)

## 5. Resource Budget Envelope (Current 5000 GPUh)

Assumption: LUMI full node = 8 allocated GPU units.
Thus `node-hours = GPUh / 8`.

Current ceiling:
- `5000 GPUh` -> `625 node-hours`

Example campaign budget:
1. Smoke campaign (8 runs x 0.5h x 1 node x 8 GPU) ~= `32 GPUh`
2. Strong scaling (1,2,4,8,16 nodes; 2h each) ~= `496 GPUh`
3. Weak scaling (1,2,4,8,16 nodes; 2h each) ~= `496 GPUh`
4. Repeat strong+weak once for confidence ~= `992 GPUh`
5. Production-like pilots:
- 3 runs x 16 nodes x 8h x 8 GPU/node ~= `3072 GPUh`

Total above ~= `4092 GPUh` (+ ~15% contingency -> `4706 GPUh`)

This fits within `5000 GPUh` and leaves some margin.

CPU-hours check at full-node layout:
- 56 cores/node * 625 node-hours = `35000 core-hours` << `500000` available
- GPU-hours are the limiting resource.

## 6. Immediate Execution Plan

1. Run smoke tests and collect diagnostics metadata.
2. Run strong/weak matrices using `slurm/lumi_g_submit_backbone_scaling_matrix.sh`.
3. Export metrics per job to a single CSV/JSON report.
4. Draft sections `3.2.4.*`, `3.2.6.1`, `3.8`, `3.9` directly from measured data.

## 7. Risks and Mitigation

Risk: environment drift between Puhti CUDA stack and LUMI ROCm stack.
Mitigation: pinned LUMI module stack + minimal extra pip layer.

Risk: queue variability at higher node counts.
Mitigation: begin with 1-8 nodes, then request larger allocations after stable baseline.

Risk: non-portable CUDA-specific assumptions in legacy launchers.
Mitigation: use LUMI-specific scripts (`slurm/lumi_g_*.sh`) for all LUMI runs.

## 8. Pixel-space vs Latent-space Scaling Recommendation

For this application, use a mixed strategy:

1. Primary scaling evidence: latent backbone training (strong + weak scaling)
- This is the actual target production workload for uncertainty-aware surrogate runs.
- It should consume most benchmark budget.

2. Secondary evidence: pixel-space AE / latent-export workflow
- Run limited tests to show end-to-end data pipeline feasibility on LUMI.
- Do not spend most GPU hours on full pixel-space scaling unless it is a core production path.

Rationale:
- Latent backbone is the dominant scientific throughput path in the proposed workflow.
- Pixel-space export is important for provenance and pipeline readiness, but often more I/O-bound.
