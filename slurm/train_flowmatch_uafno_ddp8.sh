#!/bin/bash
#SBATCH --job-name=flowmatch_uafno_ddp8
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # torchrun will launch 4 procs per node (1 per GPU)
#SBATCH --cpus-per-task=40           # ~10 CPU cores per GPU
#SBATCH --gres=gpu:v100:4            # Puhti GPU nodes have 4×V100
#SBATCH --mem=320G                   # full node memory
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=2             # keep OMP small; plenty of cores reserved
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export GIT_PYTHON_REFRESH=quiet

PY=/scratch/project_2008261/physics_ml/bin/python3.11
TRAIN=/scratch/project_2008261/pf_surrogate_modelling/models/train/core/train.py
CFG=/scratch/project_2008261/pf_surrogate_modelling/configs/train/train_flowmatch_uafno_bottleneck_d8.yaml

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1 {print $1}')
MASTER_PORT="${MASTER_PORT:-29400}"

echo "=== Launching DDP: $CFG on ${SLURM_JOB_NUM_NODES} nodes x 4 GPUs/node ==="
srun --cpu-bind=cores --hint=nomultithread \
  "$PY" -m torch.distributed.run \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    "$TRAIN" -c "$CFG"
