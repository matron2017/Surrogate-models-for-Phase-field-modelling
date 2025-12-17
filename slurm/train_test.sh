#!/bin/bash
#SBATCH --job-name=rs_train_test
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

# Python interpreter
PY=/scratch/project_2008261/physics_ml/bin/python3.11
ROOT=/scratch/project_2008261/pf_surrogate_modelling

# Paths
TRAIN_SCRIPT=${ROOT}/models/train/core/train.py
CFG=${ROOT}/configs/train/test_train.yaml

# Threading for BLAS (cap CPU threads at 10 per GPU)
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# NCCL settings suitable for short gputest runs
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING

## 4 GPUs via torch.distributed + torchrun
#srun --mem=0 --cpu-bind=cores --hint=nomultithread \
#  ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
#  ${TRAIN_SCRIPT} -c ${CFG}
##

# ---- Alternatives ----
srun --mem=0 --cpu-bind=cores --hint=nomultithread \
  ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  ${TRAIN_SCRIPT} -c ${CFG}

# To validate multi-GPU collectives, uncomment below and request --gres=gpu:v100:2.
# srun --mem=0 --cpu-bind=cores --hint=nomultithread \
#   ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
#   ${TRAIN_SCRIPT} -c ${CFG}
