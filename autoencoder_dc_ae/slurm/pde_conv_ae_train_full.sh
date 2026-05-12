#!/bin/bash
# Train custom PDEConvAE (16× spatial, 32ch, 24× total compression)
# 2 nodes × 4 GPUs = 8 GPUs  |  per-GPU batch 4  |  global batch 32
#SBATCH --job-name=pde_ae_train
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=40
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae

set -euo pipefail

PROJECT_ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

if [[ ! -f "${VENV_DIR}/bin/python3.11" ]]; then
  echo "[pde_ae] Environment not found: ${VENV_DIR}" >&2; exit 1
fi

CFG=${CFG:-${PROJECT_ROOT}/configs_current/autoencoder/train_pde_conv_ae_f16c32_512.yaml}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
RUNS_ROOT=${RUNS_ROOT:-${PROJECT_ROOT}/runs}

mkdir -p "${PROJECT_ROOT}/logs/slurm"
module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29502}

echo "======================================================="
echo " PDEConvAE training  job=${SLURM_JOB_ID}  nodes=${SLURM_NNODES}"
echo " CFG:  ${CFG}"
echo " DATA: ${DATA_ROOT}  RUNS: ${RUNS_ROOT}"
echo " master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "======================================================="

"${VENV_DIR}/bin/python3.11" --version
"${VENV_DIR}/bin/python3.11" -c "import torch; print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}')"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${VENV_DIR}/bin/torchrun" \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/train_pde_conv_ae.py" \
    --config "${CFG}"
