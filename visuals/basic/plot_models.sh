#!/bin/bash
#SBATCH --job-name=smoke_eval_2win
#SBATCH --account=project_2008261
#SBATCH --partition=small
#SBATCH --time=2:13:00
#SBATCH --cpus-per-task=64
#SBATCH --output=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/rapid_solidification/logs/slurm/%x_%j.err

set -euo pipefail

# Project paths
export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"
export PF_DATA_CONFIG="/scratch/project_2008261/rapid_solidification/configs/visuals/rapid_solid_visuals.yaml"

# Inputs for evaluation (trained run)
CFG="/scratch/project_2008261/rapid_solidification/runs/UNet_SSA_PreSkip_Full_trained/config_snapshot.yaml"
CKPT="/scratch/project_2008261/rapid_solidification/runs/UNet_SSA_PreSkip_Full_trained/checkpoint.best.pth"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="/scratch/project_2008261/rapid_solidification/runs/UNet_SSA_PreSkip_Full_trained"

# Common args
BATCH=1
VIS_CH=0
RES_PX=300000
QLOW=1.0
QHIGH=99.0

PY="/scratch/project_2008261/physics_ml/bin/python3.11"
EVAL="/scratch/project_2008261/rapid_solidification/visuals/basic/models_plots.py"

########################
# A) 281000–282000: metrics + 2 panels
########################
OUT_A="${ROOT_OUT}/eval_styled_${STAMP}_281000_282000"
${PY} ${EVAL} \
  --config "${CFG}" \
  --ckpt "${CKPT}" \
  --batch "${BATCH}" \
  --tmin 281000 --tmax 282000 \
  --dt-index -1 --dt-eq --dt-threshold \
  --vis-n 2 --vis-channel "${VIS_CH}" \
  --reservoir-px "${RES_PX}" --q-low "${QLOW}" --q-high "${QHIGH}" \
  --outdir "${OUT_A}"

########################
# B) 280000–282000: metrics only (no panels)
########################
OUT_B="${ROOT_OUT}/eval_styled_${STAMP}_280000_282000"
${PY} ${EVAL} \
  --config "${CFG}" \
  --ckpt "${CKPT}" \
  --batch "${BATCH}" \
  --tmin 280000 --tmax 282000 \
  --dt-index -1 --dt-eq --dt-threshold \
  --vis-n 0 --vis-channel "${VIS_CH}" \
  --reservoir-px "${RES_PX}" --q-low "${QLOW}" --q-high "${QHIGH}" \
  --outdir "${OUT_B}"
