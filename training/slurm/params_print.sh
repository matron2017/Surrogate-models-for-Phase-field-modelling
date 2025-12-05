#!/bin/bash
#SBATCH --job-name=params_print
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=0
#SBATCH --time=00:11:00
#SBATCH --output=%x_%j.out
set -euo pipefail


/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/training/core/number_params_models.py
