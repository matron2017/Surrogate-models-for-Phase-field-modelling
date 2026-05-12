#!/bin/bash
#SBATCH --job-name=eval_all_test
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/eval/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/eval/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
VENV=/scratch/project_2008261/physics_ml
TEST_H5=${ROOT}/autoencoder_dc_ae/data/test.h5

export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export GIT_PYTHON_REFRESH=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_SAMPLES=${N_SAMPLES:-3}
N_STEPS=${N_STEPS:-20}
OUT_DIR=${OUT_DIR:-$ROOT/eval/plots/test}

mkdir -p "$OUT_DIR"
echo "[eval_test] n_samples=$N_SAMPLES  n_steps=$N_STEPS  h5=$TEST_H5  out=$OUT_DIR"

$VENV/bin/python "$ROOT/eval/scripts/eval_all_models.py" \
    --out_dir  "$OUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --n_steps   "$N_STEPS" \
    --h5        "$TEST_H5"

echo "[eval_test] Done. Plots in $OUT_DIR"
