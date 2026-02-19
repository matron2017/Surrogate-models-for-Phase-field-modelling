#!/bin/bash
#SBATCH --job-name=gputest_bridge_best_plot
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
SCRIPT=${ROOT}/scripts/plot_bridge_endpoint_from_ckpt.py

BRIDGE_CKPT=${BRIDGE_CKPT:-/scratch/project_2008261/solidification_modelling/runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8/UNetFiLMAttn/checkpoint.best.pth}
AE_CKPT=${AE_CKPT:-/scratch/project_2008261/solidification_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth}
OUT_DIR=${OUT_DIR:-${ROOT}/results/visuals/bridge_epoch_best_current_gputest}
SPLIT=${SPLIT:-val}
INDICES=${INDICES:-234,255,300}
NFE=${NFE:-20}

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$OUT_DIR"

echo "[plot-job] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[plot-job] bridge=${BRIDGE_CKPT}"
echo "[plot-job] ae=${AE_CKPT}"
echo "[plot-job] out=${OUT_DIR}"
nvidia-smi || true

"$PY" "$SCRIPT" \
  --bridge-ckpt "$BRIDGE_CKPT" \
  --ae-ckpt "$AE_CKPT" \
  --split "$SPLIT" \
  --indices "$INDICES" \
  --bridge-nfe "$NFE" \
  --out-dir "$OUT_DIR" \
  --phase-ch 0 \
  --concentration-ch 1 \
  --concentration-scale 3 \
  --device cuda
