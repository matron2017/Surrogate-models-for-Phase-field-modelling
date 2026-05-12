#!/bin/bash
#SBATCH --job-name=dcae_compare
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/eval/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/eval/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
VENV=/scratch/project_2008261/physics_ml
DC_GEN=${ROOT}/autoencoder_dc_ae/external_refs/DC-Gen

export PYTHONPATH=${ROOT}:${DC_GEN}:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GIT_PYTHON_REFRESH=quiet

echo "[dcae_compare] Starting  job=${SLURM_JOB_ID}"
$VENV/bin/python "$ROOT/eval/scripts/dcae_pretrained_vs_finetuned.py"
echo "[dcae_compare] Done."
