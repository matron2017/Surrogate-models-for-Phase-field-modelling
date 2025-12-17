#!/bin/bash
#SBATCH --job-name=training_uafno
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=0
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

# Python interpreter
PY="/scratch/project_2008261/physics_ml/bin/python3.11"

# Paths
ROOT=/scratch/project_2008261/pf_surrogate_modelling
TRAIN_SCRIPT=${ROOT}/models/train/core/train.py
CFG=${ROOT}/configs/train/uafno.yaml

# Threading for BLAS
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# NCCL settings suitable for short gputest runs
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_ASYNC_ERROR_HANDLING

# 4 GPUs via torch.distributed + torchrun
srun --mem=0 --cpu-bind=cores --hint=nomultithread \
  ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  ${TRAIN_SCRIPT} -c ${CFG}
#

# ---- Alternatives ----
# 2 GPUs (set --cpus-per-task=20; keep OMP_NUM_THREADS=10 to stay within 10 cores/GPU)
#srun --mem=0 --cpu-bind=cores --hint=nomultithread \
#  ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
#  ${TRAIN_SCRIPT} -c ${CFG}
##
#1 GPU (set --cpus-per-task=10 and OMP_NUM_THREADS=10)
#srun --mem=0 --cpu-bind=cores --hint=nomultithread \
#   ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
#   ${TRAIN_SCRIPT} -c ${CFG}
#
