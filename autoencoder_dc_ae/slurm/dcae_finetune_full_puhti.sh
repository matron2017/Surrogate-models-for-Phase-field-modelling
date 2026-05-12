#!/bin/bash
# Full DC-AE fine-tune training on PUHTI (NVIDIA GPU).
# Trains on full dataset with checkpoint saving.
# Usage: sbatch slurm/dcae_finetune_full_puhti.sh

#SBATCH --job-name=dcae_ft_full_puhti
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=7
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${PROJECT_ROOT}"

# Use pre-configured PUHTI unified environment
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}

if [[ ! -f "${VENV_DIR}/bin/python3.11" ]]; then
  echo "[train] Environment not found: ${VENV_DIR}" >&2
  echo "[train] Run: bash scripts/setup_puhti_unified_env.sh --execute" >&2
  exit 1
fi

CFG=${CFG:-${PROJECT_ROOT}/configs_current/autoencoder/finetune/dc_ae_f32c32_pde_512.yaml}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
TMP_ROOT=${TMP_ROOT:-${PROJECT_ROOT}/tmp}
RUNS_ROOT=${RUNS_ROOT:-${PROJECT_ROOT}/runs}
DC_GEN_REPO_ROOT=${DC_GEN_REPO_ROOT:-${PROJECT_ROOT}/external_refs/DC-Gen}
export PYTHONPATH=${PROJECT_ROOT}:${DC_GEN_REPO_ROOT}:${PYTHONPATH:-}

mkdir -p "${PROJECT_ROOT}/logs/slurm"

echo "======================================================="
echo " DC-AE fine-tune FULL TRAINING (PUHTI/NVIDIA)"
echo " JOB_ID : ${SLURM_JOB_ID}"
echo " GPU    : ${CUDA_VISIBLE_DEVICES:-0}"
echo "======================================================="
echo "PROJECT_ROOT  : ${PROJECT_ROOT}"
echo "VENV_DIR      : ${VENV_DIR}"
echo "DATA_ROOT     : ${DATA_ROOT}"
echo "CONFIG        : ${CFG}"
echo "======================================================="

# Load PUHTI modules if available
module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true

# Verify environment
echo "[setup] Checking Python environment..."
"${VENV_DIR}/bin/python3.11" --version

echo "[setup] Checking PyTorch..."
"${VENV_DIR}/bin/python3.11" -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Run training
echo "[train] Starting DC-AE fine-tune full training..."
"${VENV_DIR}/bin/python3.11" \
    "${PROJECT_ROOT}/scripts/train_dcae_finetune.py" \
    --config "${CFG}"

TRAIN_EXIT=$?

echo "======================================================="
if [[ ${TRAIN_EXIT} -eq 0 ]]; then
    echo " FULL TRAINING COMPLETED SUCCESSFULLY"
else
    echo " FULL TRAINING FAILED (exit code: ${TRAIN_EXIT})"
fi
echo "======================================================="
exit ${TRAIN_EXIT}
