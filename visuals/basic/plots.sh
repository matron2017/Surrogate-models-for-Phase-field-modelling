#!/bin/bash
#SBATCH --job-name=curv_gpu_slice
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --gres=gpu:a100_1g.5gb:1     # one A100 slice
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out

# Fixed memory (17.5 GiB) is implied by the slice; do not set --mem.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nvidia-smi                           # sanity check: slice visible

# Optional: quick CUDA check for PyTorch
/scratch/project_2008261/physics_ml/bin/python3.11 - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("device:", torch.cuda.get_device_name(0))
PY

# Run workload
#/scratch/project_2008261/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/training/core/train.py
/scratch/project_2008261/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/visuals/basic/curvature_plot.py
