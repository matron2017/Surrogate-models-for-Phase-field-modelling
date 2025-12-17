#!/bin/bash
#SBATCH --job-name=flowmatch_sweep
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

PY=/scratch/project_2008261/physics_ml/bin/python3.11
TRAIN=/scratch/project_2008261/pf_surrogate_modelling/models/train/core/train.py

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export GIT_PYTHON_REFRESH=quiet

run_cfg() {
  local cfg="$1"
  echo "=== Running $cfg ==="
  srun --cpu-bind=cores --hint=nomultithread "$PY" "$TRAIN" -c "$cfg"
}

run_cfg /scratch/project_2008261/pf_surrogate_modelling/configs/train/train_flowmatch_unet_smoke.yaml
run_cfg /scratch/project_2008261/pf_surrogate_modelling/configs/train/train_flowmatch_uafno_smoke.yaml
run_cfg /scratch/project_2008261/pf_surrogate_modelling/configs/train/train_ads_convnext_gputest.yaml
run_cfg /scratch/project_2008261/pf_surrogate_modelling/configs/train/train_flowmatch_uvit_smoke.yaml
run_cfg /scratch/project_2008261/pf_surrogate_modelling/configs/train/train_diffusion_cosine_smoke.yaml
