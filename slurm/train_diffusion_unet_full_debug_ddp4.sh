#!/bin/bash
#SBATCH --job-name=diffusion_unet_full_debug_ddp4
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun spawns 4 ranks on the node
#SBATCH --cpus-per-task=40          # 10 cores per GPU
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export GIT_PYTHON_REFRESH=quiet

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH=${ROOT}/models
PY=/scratch/project_2008261/physics_ml/bin/python3.11
CFG=${ROOT}/configs/train/train_diffusion_unet_full_debug.yaml

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1 {print $1}')
MASTER_PORT="${MASTER_PORT:-29501}"

echo "=== Launching 4-GPU diffusion UNet (debug, gputest) ==="
srun --cpu-bind=cores --hint=nomultithread \
  "$PY" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    -m models.train.core.train -c "$CFG"
