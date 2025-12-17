#!/bin/bash
#SBATCH --job-name=flowmatch_uafno_smoke_2g
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # torchrun will launch 2 procs on this node
#SBATCH --cpus-per-task=20           # ~10 cores per GPU
#SBATCH --gres=gpu:v100:2
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export GIT_PYTHON_REFRESH=quiet

PY=/scratch/project_2008261/physics_ml/bin/python3.11
TRAIN=/scratch/project_2008261/pf_surrogate_modelling/models/train/core/train.py
CFG=/scratch/project_2008261/pf_surrogate_modelling/configs/train/train_flowmatch_uafno_smoke.yaml

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1 {print $1}')
MASTER_PORT="${MASTER_PORT:-29400}"

echo "=== Launching 2-GPU stochastic smoke UAFNO (3% subset) ==="
srun --cpu-bind=cores --hint=nomultithread \
  "$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    "$TRAIN" -c "$CFG"
