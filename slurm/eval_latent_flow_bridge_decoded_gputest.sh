#!/bin/bash
# Quick decoded diagnostics for latent flow + bridge checkpoints on gputest.

#SBATCH --job-name=gputest_latent_decoded_diag
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling
PY=/scratch/project_2008261/physics_ml/bin/python3
SCRIPT=${ROOT}/scripts/eval_latent_flow_bridge_decoded.py

FLOW_CKPT=${FLOW_CKPT:-${ROOT}/runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic/UNetFiLMAttn/checkpoint.best.pth}
BRIDGE_CKPT=${BRIDGE_CKPT:-}
AE_CKPT=${AE_CKPT:-${ROOT}/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth}

SPLIT=${SPLIT:-val}
MAX_BATCHES=${MAX_BATCHES:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
FLOW_NFE=${FLOW_NFE:-4}
BRIDGE_T_INDEX=${BRIDGE_T_INDEX:-128}
FLOW_NUM_SAMPLES=${FLOW_NUM_SAMPLES:-4}
FLOW_NOISE_STD=${FLOW_NOISE_STD:--1.0}
FLOW_NOISE_MODE=${FLOW_NOISE_MODE:-scalar}
FLOW_NOISE_PERTURB_SOURCE=${FLOW_NOISE_PERTURB_SOURCE:-1}

OUT_DIR=${OUT_DIR:-${ROOT}/results}
FLOW_OUT=${OUT_DIR}/diag_flow_decoded_${SPLIT}_job${SLURM_JOB_ID}.json
BRIDGE_OUT=${OUT_DIR}/diag_bridge_decoded_${SPLIT}_job${SLURM_JOB_ID}.json

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING=FALSE

echo "[diag] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[diag] flow_ckpt=${FLOW_CKPT}"
echo "[diag] bridge_ckpt=${BRIDGE_CKPT}"
echo "[diag] ae_ckpt=${AE_CKPT}"
nvidia-smi || true

cd "${ROOT}"

FLOW_NOISE_ARGS=()
if [[ "${FLOW_NOISE_PERTURB_SOURCE}" != "0" && "${FLOW_NOISE_PERTURB_SOURCE}" != "false" && "${FLOW_NOISE_PERTURB_SOURCE}" != "False" ]]; then
  FLOW_NOISE_ARGS+=(--flow-noise-perturb-source)
fi

"${PY}" "${SCRIPT}" \
  --model-ckpt "${FLOW_CKPT}" \
  --ae-ckpt "${AE_CKPT}" \
  --mode flow_rollout \
  --split "${SPLIT}" \
  --device cuda \
  --max-batches "${MAX_BATCHES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 0 \
  --flow-nfe "${FLOW_NFE}" \
  --flow-num-samples "${FLOW_NUM_SAMPLES}" \
  --flow-noise-std "${FLOW_NOISE_STD}" \
  --flow-noise-mode "${FLOW_NOISE_MODE}" \
  "${FLOW_NOISE_ARGS[@]}" \
  --out-json "${FLOW_OUT}"

if [[ -n "${BRIDGE_CKPT}" ]]; then
  "${PY}" "${SCRIPT}" \
    --model-ckpt "${BRIDGE_CKPT}" \
    --ae-ckpt "${AE_CKPT}" \
    --mode bridge_teacher_forced \
    --split "${SPLIT}" \
    --device cuda \
    --max-batches "${MAX_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 0 \
    --bridge-t-index "${BRIDGE_T_INDEX}" \
    --out-json "${BRIDGE_OUT}"
else
  echo "[diag] BRIDGE_CKPT not set; skipping bridge diagnostic"
fi

echo "[diag] wrote:"
echo "  ${FLOW_OUT}"
if [[ -n "${BRIDGE_CKPT}" ]]; then
  echo "  ${BRIDGE_OUT}"
fi
