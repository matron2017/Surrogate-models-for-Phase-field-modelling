#!/bin/bash
# DC-AE fine-tune  —  16 GPUs (4 nodes × 4 V100), global effective batch = 128 (bs=4/GPU × accum=2)
# Resumes from dc_ae_f32c32_lr2e5 checkpoint with fresh cosine LR schedule.
#SBATCH --job-name=dcae_bs128
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae

set -euo pipefail

PROJECT_ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/external_refs/DC-Gen:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

CFG=${CFG:-${PROJECT_ROOT}/configs_current/autoencoder/finetune/dc_ae_f32c32_pde_512_bs128_16gpu.yaml}
export DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
export RUNS_ROOT=${RUNS_ROOT:-${PROJECT_ROOT}/runs}

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29503}

# NCCL settings that work on Puhti InfiniBand fabric
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

mkdir -p "${PROJECT_ROOT}/logs/slurm"

echo "======================================================="
echo " DC-AE bs=128 (16 GPUs)  job=${SLURM_JOB_ID}"
echo " nodes=${SLURM_NNODES}  gpus/node=4  bs=4/gpu  accum=2  eff_global=128"
echo " config=${CFG}"
echo "======================================================="

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${VENV_DIR}/bin/torchrun" \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/train_dcae_finetune.py" \
    --config "${CFG}"

echo "DC-AE bs=128 training DONE"
