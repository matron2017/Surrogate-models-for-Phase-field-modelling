#!/bin/bash
#SBATCH --job-name=smoke_train
#SBATCH --account=project_2008261
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

set -euo pipefail

export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"
export PF_DATA_CONFIG=/scratch/project_2008261/rapid_solidification/configs/data/phase_field_data.yaml

/scratch/project_2008261/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/models/datapipes/build_hdf5_datasets.py
