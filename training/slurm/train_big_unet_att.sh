#!/bin/bash
#SBATCH --job-name=training_unet_att
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=0
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# Python interpreter
PY="/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11"

# Paths
TRAIN_SCRIPT=/scratch/project_2008261/rapid_solidification/training/core/train.py
CFG=/scratch/project_2008261/rapid_solidification/configs/train_model/rapid_solidification/unetssa.yaml

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
# 2 GPUs (halve cpus-per-task to 32; set OMP_NUM_THREADS=16)
#srun --mem=0 --cpu-bind=cores --hint=nomultithread \
#  ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
#  ${TRAIN_SCRIPT} -c ${CFG}
##
#1 GPU (cpus-per-task can remain 16–32 for dataloader headroom)
#srun --mem=0 --cpu-bind=cores --hint=nomultithread \
#   ${PY} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
#   ${TRAIN_SCRIPT} -c ${CFG}
#
