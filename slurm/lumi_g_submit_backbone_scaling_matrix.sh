#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}

ACCOUNT=${ACCOUNT:-project_462001306}
CFG_BASE=${CFG_BASE:-${PROJECT_ROOT}/configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b80_rdbmres_afno8_stochastic.yaml}
TRAIN_H5=${TRAIN_H5:-${PROJECT_ROOT}/data/latent_best_psgd_e279_dev/train_latent_experimental_midtrain.h5}
VAL_H5=${VAL_H5:-${PROJECT_ROOT}/data/latent_best_psgd_e279_dev/val_latent_experimental_midtrain.h5}
SIM_MAP=${SIM_MAP:-${PROJECT_ROOT}/data/stochastic/sim_map.json}

# Space-separated node counts.
STRONG_NODES=${STRONG_NODES:-"1 2 4 8"}
WEAK_NODES=${WEAK_NODES:-"1 2 4 8"}

GLOBAL_BATCH=${GLOBAL_BATCH:-80}
WEAK_BATCH_PER_GPU=${WEAK_BATCH_PER_GPU:-1}
WEAK_ACCUM_STEPS=${WEAK_ACCUM_STEPS:-1}

EPOCHS=${EPOCHS:-3}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-80}
NUM_WORKERS=${NUM_WORKERS:-2}

TIME_LIMIT=${TIME_LIMIT:-01:00:00}
RUN_TAG=${RUN_TAG:-lumi_flow_scale}
DRY_RUN=${DRY_RUN:-0}

submit_one() {
  local mode=$1
  local nodes=$2
  local partition="small-g"
  if (( nodes > 4 )); then
    partition="standard-g"
  fi

  local job_name="${RUN_TAG}_${mode}_n${nodes}"
  local export_args
  if [[ "${mode}" == "strong" ]]; then
    export_args="ALL,PROJECT_ROOT=${PROJECT_ROOT},CFG_BASE=${CFG_BASE},TRAIN_H5=${TRAIN_H5},VAL_H5=${VAL_H5},SIM_MAP=${SIM_MAP},MODE=strong,GLOBAL_BATCH=${GLOBAL_BATCH},BATCH_PER_GPU=1,EPOCHS=${EPOCHS},STEPS_PER_EPOCH=${STEPS_PER_EPOCH},NUM_WORKERS=${NUM_WORKERS},RUN_TAG=${RUN_TAG}"
  else
    export_args="ALL,PROJECT_ROOT=${PROJECT_ROOT},CFG_BASE=${CFG_BASE},TRAIN_H5=${TRAIN_H5},VAL_H5=${VAL_H5},SIM_MAP=${SIM_MAP},MODE=weak,BATCH_PER_GPU=${WEAK_BATCH_PER_GPU},WEAK_ACCUM_STEPS=${WEAK_ACCUM_STEPS},EPOCHS=${EPOCHS},STEPS_PER_EPOCH=${STEPS_PER_EPOCH},NUM_WORKERS=${NUM_WORKERS},RUN_TAG=${RUN_TAG}"
  fi

  local cmd=(
    sbatch
    --account="${ACCOUNT}"
    --partition="${partition}"
    --nodes="${nodes}"
    --ntasks-per-node=8
    --gpus-per-node=8
    --cpus-per-task=7
    --time="${TIME_LIMIT}"
    --job-name="${job_name}"
    --export="${export_args}"
    "${SCRIPT_DIR}/lumi_g_backbone_scaling_job.sh"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

for n in ${STRONG_NODES}; do
  submit_one strong "${n}"
done

for n in ${WEAK_NODES}; do
  submit_one weak "${n}"
done
