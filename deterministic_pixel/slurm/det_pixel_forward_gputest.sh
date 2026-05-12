#!/bin/bash
#SBATCH --job-name=det_px_fwd
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel
set -euo pipefail
ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
DET=$ROOT/deterministic_pixel
VENV=/scratch/project_2008261/physics_ml
export PYTHONPATH=/scratch/project_2008261/pf_surrogate_modelling:${PYTHONPATH:-}
OUT=$DET/tmp/forward_gputest_${SLURM_JOB_ID}
mkdir -p "$OUT"
$VENV/bin/python "$DET/scripts/test_det_pixel_forward.py" \
  --config "$DET/configs_current/pf_surrogates/smoke/train_det_unet_afno_controlxs_wavelet_512_smoke.yaml" \
  --device cuda \
  --split val \
  --out-json "$OUT/forward_report.json"
