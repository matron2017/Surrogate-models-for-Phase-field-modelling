#!/bin/bash
# 12-hour latent diffusion-bridge run (1 node x 4 GPUs), predict-next objective.

#SBATCH --job-name=latent_unet_bridge12h_predictnext_1n4g
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export CFG=${CFG:-${ROOT}/configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu7h_ddp8_b80_controlhint_predictnext.yaml}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}

exec "${ROOT}/slurm/train_unet_bridge_latent_gpu7h_ddp8_b80.sh"
