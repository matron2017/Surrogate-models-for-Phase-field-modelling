#!/bin/bash
# Flow-only best-checkpoint endpoint visuals on gputest.

#SBATCH --job-name=gputest_flow_best_only
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
SCRIPT=${ROOT}/scripts/plot_flow_best_only.py

FLOW_CKPT=${FLOW_CKPT:-${ROOT}/runs/flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic/UNetFiLMAttn/checkpoint.best.pth}
AE_CKPT=${AE_CKPT:-${ROOT}/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth}
H5_OVERRIDE=${H5_OVERRIDE:-${ROOT}/data/latent_best_psgd_e279_dev/val_latent_experimental_midtrain.h5}
INDICES=${INDICES:-0,255,300}
FLOW_NFE=${FLOW_NFE:-20}
OUT_DIR=${OUT_DIR:-${ROOT}/results/visuals/flow_best_only}
SCALE_BATCH_SIZE=${SCALE_BATCH_SIZE:-32}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING=FALSE

echo "[flow_best_only] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[flow_best_only] flow_ckpt=${FLOW_CKPT}"
echo "[flow_best_only] ae_ckpt=${AE_CKPT}"
echo "[flow_best_only] h5_override=${H5_OVERRIDE}"
echo "[flow_best_only] indices=${INDICES}"
nvidia-smi || true

cd "${ROOT}"

"${PY}" "${SCRIPT}" \
  --flow-ckpt "${FLOW_CKPT}" \
  --ae-ckpt "${AE_CKPT}" \
  --h5-override "${H5_OVERRIDE}" \
  --indices "${INDICES}" \
  --flow-nfe "${FLOW_NFE}" \
  --scale-batch-size "${SCALE_BATCH_SIZE}" \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --clean
