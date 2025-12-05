#!/bin/bash
#SBATCH --job-name=curv_plot
#SBATCH --account=project_2008261
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=nvme:200

set -euo pipefail

# Project paths
export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"

# Point the preprocessor to the YAML config
export PF_DATA_CONFIG='/scratch/project_2008261/rapid_solidification/configs/visuals/rapid_solid_visuals.yaml'

# Run
/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 \
  /scratch/project_2008261/rapid_solidification/visuals/basic/curvature_plot.py
