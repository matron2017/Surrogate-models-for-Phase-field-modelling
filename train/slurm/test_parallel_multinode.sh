#!/bin/bash
# 2 nodes × 4 GPUs (Puhti gpu partition) multi-node DDP sanity check + smoke train.
# Mirrors CSC's recommended rendezvous setup (torch.distributed.run with --rdzv_*).
# Requires project_2008261 to have quota on the gpu partition.
#
# 1. Validates NCCL connectivity via train/core/ddp_multi_node_check.py.
# 2. Launches train/core/parallel_solid_data.py on all 8 GPUs using the placeholder HDF5 data.

#SBATCH --job-name=rs_ddp_2x4
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:15:00
#SBATCH --mem=0
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

PYTHON_BIN="/scratch/project_2008261/physics_ml/bin/python3.11"
ROOT=/scratch/project_2008261/pf_surrogate_modelling
GPUS_PER_NODE=4
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1 {print $1}')
MASTER_PORT="${MASTER_PORT:-29400}"
RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
WORLD=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))

echo "[env] nodes=${SLURM_JOB_NUM_NODES} gpus_per_node=${GPUS_PER_NODE} world=${WORLD}"
echo "[env] master_host=${MASTER_HOST} master_ip=${MASTER_ADDR} port=${MASTER_PORT}"

common_launch=(
  srun --kill-on-bad-exit=1 --cpu-bind=cores --exact
  "$PYTHON_BIN" -m torch.distributed.run
    --nnodes="${SLURM_JOB_NUM_NODES}"
    --nproc_per_node="${GPUS_PER_NODE}"
    --rdzv_backend=c10d
    --rdzv_endpoint="${RDZV_ENDPOINT}"
)

echo "[step] DDP connectivity check"
"${common_launch[@]}" \
  --rdzv_id="${SLURM_JOB_ID}_check" \
  ${ROOT}/models/train/core/ddp_multi_node_check.py

echo "[step] Parallel solid-data smoke training"
"${common_launch[@]}" \
  --rdzv_id="${SLURM_JOB_ID}_train" \
  ${ROOT}/models/train/core/parallel_solid_data.py \
  --epochs=1 \
  --batch-size=2 \
  --num-workers=8 \
  --limit-total=64 \
  --limit-per-group=16 \
  --data-root=${ROOT}/data/deterministic \
  --pf-loader=${ROOT}/models/train/core/pf_dataloader.py \
  --pf-class=PFPairDataset

echo "[diag] seff summary"
seff "${SLURM_JOB_ID}" || true

echo "[diag] sacct summary"
sacct -j "${SLURM_JOB_ID}" \
  -o jobid,jobname,partition,allocnodes,alloccpus,elapsed,state%16 \
  -P || true

echo "[done] Multi-node DDP run completed."
