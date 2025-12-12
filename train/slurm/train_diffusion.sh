#!/bin/bash
#SBATCH --job-name=diffusion_smoke
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
CFG=${ROOT}/experiments/diffusion_prototype/configs/ddpm_placeholder.json

srun --cpu-bind=cores --hint=nomultithread "$PY" \
  ${ROOT}/experiments/diffusion_prototype/src/train_ddpm_residual.py \
  --config "$CFG"
