#!/bin/bash
#SBATCH --job-name=dcae_preplot
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
set -euo pipefail
PROJECT_ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}
export PROJECT_ROOT DATA_ROOT=${PROJECT_ROOT}/data DC_GEN_REPO_ROOT=${PROJECT_ROOT}/external_refs/DC-Gen
export PYTHONPATH=${PROJECT_ROOT}:${DC_GEN_REPO_ROOT}:${PYTHONPATH:-}
module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true
OUT=${PROJECT_ROOT}/plots/pretrained_base/${SLURM_JOB_ID}
"${VENV_DIR}/bin/python3.11" scripts/plot_dcae_pretrained_base.py --sample-h5 ${PROJECT_ROOT}/data/val.h5 --t-index 0 --stats-max-frames 200 --out-dir "$OUT"
