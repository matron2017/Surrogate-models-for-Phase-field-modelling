#!/bin/bash
# Build experimental latent datasets (train/val/test) from an in-progress AE checkpoint.

#SBATCH --job-name=export_latent_best_psgd
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

CHECKPOINT=${CHECKPOINT:-${ROOT}/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth}
OUT_DIR=${OUT_DIR:-${ROOT}/data/latent_best_psgd_e279_dev}
MAX_GROUPS=${MAX_GROUPS:-0}

echo "[latent-export] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[latent-export] checkpoint=${CHECKPOINT}"
echo "[latent-export] out_dir=${OUT_DIR}"
echo "[latent-export] max_groups=${MAX_GROUPS} (0 means full split)"
nvidia-smi || true

cd "${ROOT}"
"${PY}" scripts/export_latent_dataset_midtrain.py \
  --checkpoint "${CHECKPOINT}" \
  --source train=${ROOT}/data/stochastic/rightclean/simulation_train_rightclean_fixed34_gradshared.h5 \
  --source val=${ROOT}/data/stochastic/rightclean/simulation_val_rightclean_fixed34_gradshared.h5 \
  --source test=${ROOT}/data/stochastic/rightclean/simulation_test_rightclean_fixed34_gradshared.h5 \
  --out-dir "${OUT_DIR}" \
  --batch-size 2 \
  --dtype float16 \
  --compression gzip \
  --compression-level 4 \
  --max-groups "${MAX_GROUPS}" \
  --device cuda \
  --amp-dtype fp16
