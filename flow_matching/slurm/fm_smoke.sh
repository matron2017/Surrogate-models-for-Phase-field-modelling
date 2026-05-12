#!/bin/bash
# Flow Matching GPU smoke test — gputest partition, 1 V100, 15 min
# Tests: model load, forward pass, Euler ODE inference, backward pass, memory
#SBATCH --job-name=fm_smoke
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/flow_matching/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/flow_matching/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}
PY="${VENV_DIR}/bin/python"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

H5=${H5:-${ROOT}/autoencoder_dc_ae/data/train.h5}
OUT_DIR=${OUT_DIR:-${ROOT}/flow_matching/runs/smoke}
N_STEPS=${N_STEPS:-20}

mkdir -p "${ROOT}/flow_matching/logs/slurm"
mkdir -p "${OUT_DIR}"

echo "======================================================="
echo " FM smoke  job=${SLURM_JOB_ID}  node=$(hostname)"
echo " n_steps=${N_STEPS}  backbone=300M+"
echo "======================================================="

"${PY}" --version
"${PY}" -c "import torch; print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}  GPU={torch.cuda.get_device_name(0)}')"

"${PY}" \
  "${ROOT}/flow_matching/scripts/smoke_fm.py" \
  --n_steps "${N_STEPS}" \
  --h5 "${H5}" \
  --out_dir "${OUT_DIR}" \
  --batch_size 2

echo "FM smoke DONE"
