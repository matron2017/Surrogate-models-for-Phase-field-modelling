#!/bin/bash
# DC-AE fine-tune resumable training — LR variant A: lr=5e-6
# Resumes from checkpoint.last.pth; epochs=1000 in config.
# Usage: sbatch slurm/dcae_finetune_resume_lr5e6.sh

#SBATCH --job-name=dcae_lr5e6
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=7
#SBATCH --time=24:00:00
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
cd "${PROJECT_ROOT}"

VENV_DIR=/scratch/project_2008261/physics_ml
DC_GEN_REPO_ROOT=${PROJECT_ROOT}/external_refs/DC-Gen

export PYTHONPATH=${PROJECT_ROOT}:${DC_GEN_REPO_ROOT}:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export GIT_PYTHON_REFRESH=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
RUNS_ROOT=${RUNS_ROOT:-${PROJECT_ROOT}/runs}
CFG=${CFG:-${PROJECT_ROOT}/configs_current/autoencoder/finetune/dc_ae_f32c32_pde_512_lr5e6.yaml}

mkdir -p "${PROJECT_ROOT}/logs/slurm"

module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true

echo "======================================================="
echo " DC-AE lr5e6  RESUME TRAINING   JOB=${SLURM_JOB_ID}"
echo " CONFIG: ${CFG}"
echo "======================================================="

"${VENV_DIR}/bin/python3.11" \
    "${PROJECT_ROOT}/scripts/train_dcae_finetune.py" \
    --config "${CFG}"

echo "[dcae_lr5e6] Done."
