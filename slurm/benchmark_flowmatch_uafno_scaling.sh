#!/bin/bash
#SBATCH --job-name=flowmatch_uafno_scale
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40           # ~10 CPU cores per GPU
#SBATCH --gres=gpu:v100:4            # 4 GPUs per node on Puhti
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
CFG=/scratch/project_2008261/pf_surrogate_modelling/configs/train/train_flowmatch_uafno_bottleneck_d8.yaml

HOSTS=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_ADDR=$(getent ahostsv4 "${HOSTS[0]}" | awk 'NR==1 {print $1}')
MASTER_PORT="${MASTER_PORT:-29400}"

# (nodes, procs_per_node) tuples to test: 1,2,4,8 GPUs
RUNS=(
  "1 1"
  "1 2"
  "1 4"
  "2 4"
)

for run in "${RUNS[@]}"; do
  read -r NNODES NPROC <<<"$run"
  echo "=== Scaling run: nnodes=${NNODES} nproc_per_node=${NPROC} (total GPUs = $((NNODES * NPROC))) ==="
  srun --nodes="${NNODES}" --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:v100:4 \
    --cpu-bind=cores --hint=nomultithread \
    "$PY" -m torch.distributed.run \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      --rdzv_id="${SLURM_JOB_ID}_${NNODES}x${NPROC}" \
      "$TRAIN" -c "$CFG"
done
