#!/usr/bin/env bash
set -euo pipefail
STACK_ROOT=/scratch/project_462001338/pf_bridge_cleanstack
PF_ROOT=/scratch/project_462001338/pf_surrogate_modelling
CFG_BASE=${CFG_BASE:-$STACK_ROOT/configs/pf_surrogates/big/train_det_unet_afno_controlxs_wavelet_512_big.yaml}
PARTITION=${PARTITION:-small-g}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-28}
TIME_LIMIT=${TIME_LIMIT:-1-00:00:00}
RUN_TASKS=${RUN_TASKS:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-8}
BATCH_PER_GPU=${BATCH_PER_GPU:-1}
EPOCHS=${EPOCHS:-260}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-80}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_TAG=${RUN_TAG:-big_det_unet_afno_controlxs_wavelet_512_${STAMP}}
cd "$PF_ROOT"
sbatch -A project_462001338 -p "$PARTITION" -N "$NODES" --ntasks-per-node=1 --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" -t "$TIME_LIMIT" -J det_afno_big \
  --export=ALL,RUN_TAG="$RUN_TAG",CFG_BASE="$CFG_BASE",MODE=strong,RUN_TASKS="$RUN_TASKS",GLOBAL_BATCH="$GLOBAL_BATCH",BATCH_PER_GPU="$BATCH_PER_GPU",EPOCHS="$EPOCHS",STEPS_PER_EPOCH="$STEPS_PER_EPOCH",USE_VAL=1,DETERMINISTIC_OVERRIDE=0,CLEAR_RESUME=1 \
  "$PF_ROOT/slurm/lumi_scale_aif_torchrun_quick.sh"
