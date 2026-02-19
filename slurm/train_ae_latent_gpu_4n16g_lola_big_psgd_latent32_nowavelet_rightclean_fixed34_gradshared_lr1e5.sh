#!/bin/bash
# AE latent full run (4 nodes x 4 GPUs, LoLA big, PSGD, rightclean fixed34 gradshared, b40).

#SBATCH --job-name=ae_latent_gpu_4n16g_lola_big_psgd_latent32_rc_lr1e5
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export GIT_PYTHON_REFRESH=quiet
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH=${ROOT}
PY=/scratch/project_2008261/physics_ml/bin/python3.11
TORCHRUN=${TORCHRUN:-$(dirname "$PY")/torchrun}
CFG=${ROOT}/configs/train/train_ae_latent_gpumedium_psgd_uncached_freq1_lola_big_64_1024_12g_latent32_nowavelet_b40.yaml

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1 {print $1}')
MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"
NODE_RANK=${SLURM_NODEID:-0}
export MASTER_ADDR MASTER_PORT

echo "=== Launching AE latent GPU (4 nodes x 4 GPUs, PSGD, rightclean fixed34, b40) ==="
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NODE_RANK=${NODE_RANK}"
srun --cpu-bind=cores --hint=nomultithread \
  "$TORCHRUN" \
    --nnodes=4 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --node_rank=${NODE_RANK} \
    -m models.train.core.train -c "$CFG"
