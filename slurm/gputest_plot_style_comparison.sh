#!/bin/bash
# Plot side-by-side model style comparison panels on gputest.

#SBATCH --job-name=gputest_style_compare
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
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING=FALSE
export CUBLAS_WORKSPACE_CONFIG=:16:8

OLD_CKPT=${OLD_CKPT:-${ROOT}/runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_predictnext_nomass_afno8_gputest_overfit_replay6e/UNetFiLMAttn/checkpoint.best.pth}
BRIDGE_CKPT=${BRIDGE_CKPT:-${ROOT}/runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8/UNetFiLMAttn/checkpoint.best.pth}
FLOW_CKPT=${FLOW_CKPT:-${ROOT}/runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic/UNetFiLMAttn/checkpoint.best.pth}
NEW_CKPT=${NEW_CKPT:-${ROOT}/runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_sfm_latent_gputest_overfit/UNetFiLMAttn/checkpoint.best.pth}
AE_CKPT=${AE_CKPT:-${ROOT}/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth}

SPLIT=${SPLIT:-val}
SAMPLE_INDICES=${SAMPLE_INDICES:-0,1}
CHANNELS=${CHANNELS:-0,1}
FLOW_NFE=${FLOW_NFE:-20}
FLOW_NUM_SAMPLES=${FLOW_NUM_SAMPLES:-1}
BRIDGE_NFE=${BRIDGE_NFE:-20}
BRIDGE_ETA=${BRIDGE_ETA:-0.0}
FLOW_NOISE_STD=${FLOW_NOISE_STD:-0.0}
DATASET_FROM=${DATASET_FROM:-flow}
OUT_DIR=${OUT_DIR:-${ROOT}/results/style_compare_job${SLURM_JOB_ID}}

echo "[style-compare] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[style-compare] old_ckpt=${OLD_CKPT}"
echo "[style-compare] bridge_ckpt=${BRIDGE_CKPT}"
echo "[style-compare] flow_ckpt=${FLOW_CKPT}"
echo "[style-compare] new_ckpt=${NEW_CKPT}"
echo "[style-compare] out_dir=${OUT_DIR}"
nvidia-smi || true

cd "${ROOT}"
"${PY}" scripts/plot_style_comparison_panels.py \
  --old-ckpt "${OLD_CKPT}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --flow-ckpt "${FLOW_CKPT}" \
  --new-ckpt "${NEW_CKPT}" \
  --ae-ckpt "${AE_CKPT}" \
  --split "${SPLIT}" \
  --sample-indices "${SAMPLE_INDICES}" \
  --channels "${CHANNELS}" \
  --flow-nfe "${FLOW_NFE}" \
  --flow-num-samples "${FLOW_NUM_SAMPLES}" \
  --bridge-nfe "${BRIDGE_NFE}" \
  --bridge-eta "${BRIDGE_ETA}" \
  --flow-noise-std "${FLOW_NOISE_STD}" \
  --flow-noise-mode scalar \
  --flow-noise-perturb-source auto \
  --dataset-from "${DATASET_FROM}" \
  --out-dir "${OUT_DIR}"

echo "[style-compare] wrote:"
find "${OUT_DIR}" -maxdepth 1 -type f | sort
