#!/bin/bash
#SBATCH --job-name=params_print
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=0
#SBATCH --time=00:11:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err
set -euo pipefail


ROOT=/scratch/project_2008261/pf_surrogate_modelling
/scratch/project_2008261/physics_ml/bin/python3.11 ${ROOT}/models/train/core/number_params_models.py
