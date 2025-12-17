#!/bin/bash
#SBATCH --job-name=rs_visual_smoke
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

set -euo pipefail

export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"

PY=/scratch/project_2008261/physics_ml/bin/python3.11
SCRIPT=/scratch/project_2008261/rapid_solidification/visuals/basic/solid_data_visual.py
CFG=/scratch/project_2008261/rapid_solidification/configs/visuals/rapid_solid_visuals_smoke.yaml

srun --cpu-bind=cores --hint=nomultithread "$PY" "$SCRIPT" "$CFG"
