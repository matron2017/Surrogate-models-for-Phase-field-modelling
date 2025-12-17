#!/bin/bash
#SBATCH --job-name=smoke_train
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=0
#SBATCH --time=0:14:00
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

set -euo pipefail

# Project paths
export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"

# Point the preprocessor to the YAML config
export PF_DATA_CONFIG='/scratch/project_2008261/rapid_solidification/configs/visuals/rapid_solid_visuals.yaml'

# Run
/scratch/project_2008261/physics_ml/bin/python3.11 \
  /scratch/project_2008261/rapid_solidification/visuals/basic/solid_data_visual.py
