#!/bin/bash
#SBATCH --job-name=rs_train_smoke
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out

set -euo pipefail

PY=/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11
CFG=/scratch/project_2008261/rapid_solidification/configs/train_model/rapid_solidification/train_smoke.yaml
TRAIN_SCRIPT=/scratch/project_2008261/rapid_solidification/training/core/train.py

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

srun --cpu-bind=cores --hint=nomultithread \
  "$PY" "$TRAIN_SCRIPT" -c "$CFG"
