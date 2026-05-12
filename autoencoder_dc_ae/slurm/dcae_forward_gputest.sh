#!/bin/bash
#SBATCH --job-name=dcae_forward_gt
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
export PROJECT_ROOT
export DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
export OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/runs/autoencoder/forward_gputest/${SLURM_JOB_ID}}
export DC_GEN_REPO_ROOT=${DC_GEN_REPO_ROOT:-${PROJECT_ROOT}/external_refs/DC-Gen}
export PYTHONPATH=${PROJECT_ROOT}:${DC_GEN_REPO_ROOT}:${PYTHONPATH:-}
module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true
"${VENV_DIR}/bin/python3.11" tests/test_dcae_forward_shapes.py
