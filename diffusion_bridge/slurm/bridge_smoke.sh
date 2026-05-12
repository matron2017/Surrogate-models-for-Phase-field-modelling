#!/bin/bash
# Diffusion bridge smoke test — both UniDB and FracBridge, 300M+ backbone
# gputest partition: single node, 1 GPU (model loads fully on one V100-32GB)
#SBATCH --job-name=bridge_smoke_big
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/diffusion_bridge/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/diffusion_bridge/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/solidification_modelling/physics_ml}
[ -x "${VENV_DIR}/bin/python" ] || VENV_DIR=/scratch/project_2008261/physics_ml
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true

H5=${H5:-${ROOT}/autoencoder_dc_ae/data/train.h5}
OUT_DIR=${OUT_DIR:-${ROOT}/diffusion_bridge/runs/smoke_big}
N_STEPS=${N_STEPS:-20}

mkdir -p "${ROOT}/diffusion_bridge/logs/slurm"
mkdir -p "${OUT_DIR}"

echo "======================================================="
echo " Bridge smoke BIG  job=${SLURM_JOB_ID}  node=$(hostname)"
echo " bridge=both  n_steps=${N_STEPS}  backbone=300M+"
echo "======================================================="

"${VENV_DIR}/bin/python" --version
"${VENV_DIR}/bin/python" -c "import torch; print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}  GPU={torch.cuda.get_device_name(0)}')"

"${VENV_DIR}/bin/python" \
  "${ROOT}/diffusion_bridge/scripts/smoke_bridge.py" \
  --bridge both \
  --n_steps "${N_STEPS}" \
  --h5 "${H5}" \
  --out_dir "${OUT_DIR}" \
  --batch_size 2

echo "Bridge smoke BIG DONE"
