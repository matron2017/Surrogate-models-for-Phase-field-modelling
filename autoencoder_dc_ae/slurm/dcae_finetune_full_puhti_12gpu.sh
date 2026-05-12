#!/bin/bash
#SBATCH --job-name=dcae_ft_full_8gpu
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae

set -euo pipefail
PROJECT_ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/autoencoder_dc_ae
cd "${PROJECT_ROOT}"
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/physics_ml}
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/external_refs/DC-Gen:${PYTHONPATH:-}
if [[ ! -f "${VENV_DIR}/bin/python3.11" ]]; then
  echo "[full8] Environment not found: ${VENV_DIR}" >&2
  exit 1
fi

CFG=${CFG:-${PROJECT_ROOT}/configs_current/autoencoder/finetune/dc_ae_f32c32_pde_512_varA.yaml}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
TMP_ROOT=${TMP_ROOT:-${PROJECT_ROOT}/tmp}
RUNS_ROOT=${RUNS_ROOT:-${PROJECT_ROOT}/runs}
DC_GEN_REPO_ROOT=${DC_GEN_REPO_ROOT:-${PROJECT_ROOT}/external_refs/DC-Gen}

mkdir -p "${PROJECT_ROOT}/logs/slurm"
module load CUDA/12.2.0 2>/dev/null || true
module load cuDNN/8.9.4.25-CUDA-12.2.0 2>/dev/null || true

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  "${VENV_DIR}/bin/torchrun" \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/train_dcae_finetune.py" \
    --config "${CFG}"
