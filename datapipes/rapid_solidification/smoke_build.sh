#!/bin/bash
#SBATCH --job-name=rs_build_smoke
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out

set -euo pipefail

export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"
export PF_DATA_CONFIG=/scratch/project_2008261/rapid_solidification/configs/data/phase_field_data_smoke.yaml

PY=/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11
SCRIPT=/scratch/project_2008261/rapid_solidification/datapipes/rapid_solidification/build_hdf5_datasets.py

srun --cpu-bind=cores --hint=nomultithread "$PY" "$SCRIPT"
