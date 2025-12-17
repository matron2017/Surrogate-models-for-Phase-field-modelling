#!/bin/bash
#SBATCH --job-name=curv_gpu_slice
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

# Fixed memory (approx 24 GiB) with single V100; do not set --mem.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nvidia-smi                           # sanity check: device visible

# Optional: quick CUDA check for PyTorch
/scratch/project_2008261/physics_ml/bin/python3.11 - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("device:", torch.cuda.get_device_name(0))
PY

# Run workload
#/scratch/project_2008261/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/models/train/core/train.py
/scratch/project_2008261/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/visuals/basic/curvature_plot.py
