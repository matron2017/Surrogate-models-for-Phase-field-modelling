# LUMI vs Puhti: Conda, Tykky, Containers (2026-02-19)

This note compares practical environment options for this project on CSC systems.

## 1. Executive Recommendation

For this PF-surrogate project:

1. Development/debug:
- Keep fast iteration on Puhti (existing workflow).

2. LUMI benchmark and scaling runs:
- Use LUMI module stack (`Local-CSC/default + pytorch/2.5`) plus a thin venv layer.

3. Before final Extreme Scale production campaign:
- Freeze environment with Tykky or a tested container image for maximum reproducibility.

## 2. Option Comparison

## A) Plain Conda Environment

Pros:
- Familiar workflow.
- Easy package management for interactive work.

Cons on HPC:
- Many small files -> metadata overhead on parallel filesystems.
- Startup and import overhead can become significant at scale.
- Harder to guarantee reproducibility across systems if ad-hoc updates are made.

Use when:
- Local development, exploratory package testing.

## B) Tykky (CSC tool)

What it does:
- Builds/encapsulates Python env into a containerized runtime style suited for CSC systems.

Pros:
- Better HPC behavior than loose Conda env trees.
- Stronger portability/reproducibility across CSC nodes.

Cons:
- Additional packaging workflow to maintain.

Use when:
- You need stable, repeated large runs and reproducible app-level environment.

## C) Direct Singularity/Apptainer Container

Pros:
- Maximum reproducibility when image is fixed.
- Common for production HPC workflows.

Cons:
- Need image build and maintenance process.
- Need to ensure ROCm/GPU runtime compatibility and mounts are correct.

Use when:
- Final production phase and publication-grade reruns.

## 3. Practical Choice for This Project Right Now

Current best path:

1. Keep current LUMI module base:
- `module load LUMI partition/G Local-CSC/default pytorch/2.5`

2. Add only missing Python extras in venv:
- `scripts/lumi_setup_physics_ml_env.sh`

3. Use LUMI-specific launchers:
- `slurm/lumi_g_ae_smoke.sh`
- `slurm/lumi_g_backbone_scaling_job.sh`
- `slurm/lumi_g_submit_backbone_scaling_matrix.sh`

4. Once benchmark matrix is stable:
- freeze env with Tykky/container for final campaign repeatability.

## 4. Why LUMI and Puhti Differ in Practice

- LUMI-G is AMD MI250X (ROCm stack), while Puhti GPU workflows are NVIDIA/CUDA-oriented.
- Legacy Puhti scripts include CUDA/NCCL/V100 assumptions and hardcoded Puhti paths.
- LUMI scripts must use LUMI partition/account layout and ROCm-compatible runtime.

## 5. References

- CSC Tykky docs: https://docs.csc.fi/computing/containers/tykky/
- CSC software on LUMI: https://docs.csc.fi/apps/by_system/#lumi
- LUMI AI Factory containers: https://docs.lumi-supercomputer.eu/runjobs/lumi_env/LUMI_AI_Factory_containers/
- LUMI-G hardware overview: https://docs.lumi-supercomputer.eu/hardware/lumig/
- LUMI-G job parameters: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/
