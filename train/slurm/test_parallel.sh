#!/bin/bash
#SBATCH --job-name=uafno_1x2
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err
set -euo pipefail

####For 2 GPUs keep --cpus-per-task=20 and OMP_NUM_THREADS=10 (10 CPU cores/GPU).####
ROOT=/scratch/project_2008261/pf_surrogate_modelling
PY=/scratch/project_2008261/physics_ml/bin/python3.11

# Two GPUs, per-GPU batch 1 → 1×2 layout. Increase global batch via --microbatch or more steps.
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

# 4 GPU
srun --mem=0 --cpu-bind=cores --hint=nomultithread \
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  ${ROOT}/models/train/core/parallel_solid_data.py \
  --epochs=3 --batch-size=8 --H=1024 --W=1024 --limit-total=128 --num-workers=12




#srun --mem=0 --cpu-bind=cores --hint=nomultithread \
#  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
#  ${ROOT}/models/train/core/parallel_model_test.py \
#  --epochs=3 --batch-size=8 --H=1024 --W=1024 --dataset-length=128 --num-workers=12
#
