#!/bin/bash
# Multi-GPU DDP smoke test on PUHTI gputest partition.
# Uses 4 GPUs on one node to validate NCCL/DDP startup and short training.
# Usage: sbatch slurm/dcae_finetune_gputest_ddp_smoke.sh

#SBATCH --job-name=dcae_ft_gputest_ddp
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=40
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae

set -euo pipefail

PROJECT_ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
cd "${PROJECT_ROOT}"

VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/external_refs/DC-Gen:${PYTHONPATH:-}
if [[ ! -f "${VENV_DIR}/bin/python3.11" ]]; then
  echo "[gputest] Environment not found: ${VENV_DIR}" >&2
  exit 1
fi

CFG=${CFG:-${PROJECT_ROOT}/configs_current/autoencoder/finetune/dc_ae_f32c32_pde_512_gputest_ddp_smoke.yaml}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
TMP_ROOT=${TMP_ROOT:-${PROJECT_ROOT}/tmp}
RUNS_ROOT=${RUNS_ROOT:-${PROJECT_ROOT}/runs}
DC_GEN_REPO_ROOT=${DC_GEN_REPO_ROOT:-${PROJECT_ROOT}/external_refs/DC-Gen}

mkdir -p "${PROJECT_ROOT}/logs/slurm"

module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true

export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "======================================================="
echo " DC-AE DDP SMOKE TEST (PUHTI gputest, 4 GPUs)"
echo " JOB_ID      : ${SLURM_JOB_ID}"
echo " SUBMIT_DIR  : ${SLURM_SUBMIT_DIR:-N/A}"
echo " PWD         : $(pwd)"
echo " NODELIST    : ${SLURM_JOB_NODELIST}"
echo " MASTER_ADDR : ${MASTER_ADDR}:${MASTER_PORT}"
echo " CONFIG      : ${CFG}"
echo "======================================================="

srun --ntasks=1 --ntasks-per-node=1 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun --ntasks=1 --ntasks-per-node=1 \
  "${VENV_DIR}/bin/torchrun" \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/train_dcae_finetune.py" \
    --config "${CFG}"