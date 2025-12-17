#!/bin/bash
#SBATCH --job-name=rs_train_smoke
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling
PY=/scratch/project_2008261/physics_ml/bin/python3.11
CFG=${ROOT}/configs/train/train_smoke.yaml
TRAIN_SCRIPT=${ROOT}/models/train/core/train.py

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-":16:8"}

srun --cpu-bind=cores --hint=nomultithread \
  "$PY" "$TRAIN_SCRIPT" -c "$CFG"
