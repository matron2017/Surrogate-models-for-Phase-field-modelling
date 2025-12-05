#!/bin/bash
#SBATCH --job-name=smoke_train
#SBATCH --account=project_2008261
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --time=12:15:00
#SBATCH --output=%x_%j.out

set -euo pipefail

export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"
export PF_DATA_CONFIG=/scratch/project_2008261/rapid_solidification/configs/data/phase_field_data.yaml

/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/datapipes/rapid_solidification/build_hdf5_datasets.py
