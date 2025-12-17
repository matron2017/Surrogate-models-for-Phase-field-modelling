#!/bin/bash
#SBATCH --job-name=rs_wavelet_a10000b50000
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

set -euo pipefail

PY="/scratch/project_2008261/physics_ml/bin/python3.11"
SCRIPT=/scratch/project_2008261/rapid_solidification/models/datapipes/precompute_wavelet_weights.py

H5_IN=/scratch/project_2008261/rapid_solidification/data/rapid_solidification/simulation_train.h5
H5_OUT=/scratch/project_2008261/rapid_solidification/data/rapid_solidification/simulation_train_a10000b50000.wavelet.h5

# Ensure legacy wavelet_weight.py is discoverable
export PYTHONPATH="/scratch/project_2008261/solidification_modelling/scripts_legacy:/scratch/project_2008261/rapid_solidification:${PYTHONPATH:-}"

echo "Input : $H5_IN"
echo "Output: $H5_OUT"

srun --cpu-bind=cores --hint=nomultithread \
  ${PY} ${SCRIPT} \
    --h5 "$H5_IN" \
    --out "$H5_OUT" \
    --target-channels 0 1 \
    --device cuda \
    --batch-size 2 \
    --alpha 10000 \
    --beta-w 50000 \
    --J 1 \
    --wave haar \
    --mode zero \
    --theta 0.75
