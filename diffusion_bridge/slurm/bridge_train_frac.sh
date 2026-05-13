#!/bin/bash
# Fractional bridge full training — 3 nodes × 4 GPUs = 12 ranks, 300M+ backbone
#SBATCH --job-name=bridge_frac
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/diffusion_bridge/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/diffusion_bridge/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/solidification_modelling/physics_ml}
[ -x "${VENV_DIR}/bin/python" ] || VENV_DIR=/scratch/project_2008261/physics_ml
PY="${VENV_DIR}/bin/python"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-${ROOT}/diffusion_bridge/configs/frac_bridge_pde_512_big.yaml}
NODES=${SLURM_NNODES:-3}
GPUS_PER_NODE=4

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=${MASTER_PORT:-29547}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

TORCHRUN="${VENV_DIR}/bin/torchrun"

mkdir -p "${ROOT}/diffusion_bridge/logs/slurm"

echo "======================================================="
echo " FracBridge FULL  job=${SLURM_JOB_ID}"
echo " nodes=${NODES}  gpus/node=${GPUS_PER_NODE}  world=$((NODES * GPUS_PER_NODE))"
echo " master=${MASTER_HOST}:${MASTER_PORT}"
echo " config=${CONFIG}"
echo "======================================================="
nvidia-smi || true

srun --ntasks="${NODES}" --ntasks-per-node=1 \
  "${TORCHRUN}" \
    --nnodes="${NODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_HOST}:${MASTER_PORT}" \
    "${ROOT}/diffusion_bridge/scripts/train_bridge.py" \
      --config "${CONFIG}"

echo "FracBridge training DONE"
