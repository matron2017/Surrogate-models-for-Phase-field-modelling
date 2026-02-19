#!/bin/bash
# Generate systematic decoded field visuals (past/true/flow/bridge/residuals) on gputest.

#SBATCH --job-name=gputest_plot_latent_fields
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
SCRIPT=${ROOT}/scripts/plot_latent_flow_bridge_fields.py

FLOW_CKPT=${FLOW_CKPT:-${ROOT}/runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic/UNetFiLMAttn/checkpoint.best.pth}
BRIDGE_CKPT=${BRIDGE_CKPT:-${ROOT}/runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8/UNetFiLMAttn/checkpoint.best.pth}
AE_CKPT=${AE_CKPT:-${ROOT}/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth}

SPLIT=${SPLIT:-test}
MAX_SAMPLES=${MAX_SAMPLES:-3}
FLOW_NFE=${FLOW_NFE:-4}
BRIDGE_T_INDEX=${BRIDGE_T_INDEX:-128}
DPI=${DPI:-220}
OUT_DIR=${OUT_DIR:-}
INDICES=${INDICES:-}
AE_METRICS_JSON=${AE_METRICS_JSON:-}
SKIP_FIRST=${SKIP_FIRST:-0}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING=FALSE

echo "[plot] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[plot] flow_ckpt=${FLOW_CKPT}"
echo "[plot] bridge_ckpt=${BRIDGE_CKPT}"
echo "[plot] ae_ckpt=${AE_CKPT}"
nvidia-smi || true

cd "${ROOT}"

ARGS=(
  --flow-ckpt "${FLOW_CKPT}"
  --bridge-ckpt "${BRIDGE_CKPT}"
  --ae-ckpt "${AE_CKPT}"
  --split "${SPLIT}"
  --max-samples "${MAX_SAMPLES}"
  --flow-nfe "${FLOW_NFE}"
  --bridge-t-index "${BRIDGE_T_INDEX}"
  --device cuda
  --dpi "${DPI}"
)

if [[ -n "${OUT_DIR}" ]]; then
  ARGS+=(--out-dir "${OUT_DIR}")
fi
if [[ -n "${INDICES}" ]]; then
  ARGS+=(--indices "${INDICES}")
fi
if [[ -n "${AE_METRICS_JSON}" ]]; then
  ARGS+=(--ae-metrics-json "${AE_METRICS_JSON}")
fi
if [[ "${SKIP_FIRST}" == "1" ]]; then
  ARGS+=(--skip-first)
fi

"${PY}" "${SCRIPT}" "${ARGS[@]}"
