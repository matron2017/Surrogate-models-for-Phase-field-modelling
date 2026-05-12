#!/bin/bash
# Rectified Flow / Flow Matching training — 3 nodes × 4 GPUs = 12 ranks
# Identical compute budget to bridge baselines (300 epochs, global batch 72)
#SBATCH --job-name=fm_pde
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/flow_matching/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/flow_matching/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}
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
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-${ROOT}/flow_matching/configs/fm_pde_512_big.yaml}
NODES=${SLURM_NNODES:-3}
GPUS_PER_NODE=4

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=${MASTER_PORT:-29546}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

mkdir -p "${ROOT}/flow_matching/logs/slurm"

echo "======================================================="
echo " Flow Matching (Rectified Flow) FULL  job=${SLURM_JOB_ID}"
echo " nodes=${NODES}  gpus/node=${GPUS_PER_NODE}  world=$((NODES * GPUS_PER_NODE))"
echo " master=${MASTER_ADDR}:${MASTER_PORT}"
echo " config=${CONFIG}"
echo "======================================================="
nvidia-smi || true

VENV_TORCHRUN="${VENV_DIR}/bin/torchrun"

srun --ntasks="${NODES}" --ntasks-per-node=1 \
  "${VENV_TORCHRUN}" \
    --nnodes="${NODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${ROOT}/flow_matching/scripts/train_fm.py" \
      --config "${CONFIG}"

echo "Flow Matching training DONE"
