#!/bin/bash
#SBATCH --job-name=wavelet_data
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=8G
#SBATCH --time=04:14:00
#SBATCH --output=%x_%j.out

set -euo pipefail

export PYTHONPATH="/scratch/project_2008261/rapid_solidification"


/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 \
  /scratch/project_2008261/rapid_solidification/datapipes/rapid_solidification/precompute_wavelet_weights.py \
  --h5  /scratch/project_2008261/rapid_solidification/data/rapid_solidification/simulation_train.h5 \
  --out /scratch/project_2008261/rapid_solidification/data/rapid_solidification/simulation_train_a3000b9000size.wavelet.h5 \
  --target-channels 0 1 \
  --device cuda:0 \
  --batch-size 8 \
  --J 1 --wave haar --mode zero --theta 0.55 --alpha 3000.0 --beta-w 9000.0
