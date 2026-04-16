#!/usr/bin/env bash
set -euo pipefail
STACK_ROOT=/scratch/project_462001338/pf_bridge_cleanstack
PF_ROOT=/scratch/project_462001338/pf_surrogate_modelling
CFG_BASE=${CFG_BASE:-/scratch/project_462001338/pf_bridge_cleanstack/configs/pf_surrogates/big/train_bridge_unidb_unet_afno_controlxs_pixel512_big.yaml}
if [[ -z "${NO_LEGACY_WARN:-}" ]]; then
  echo "[legacy-name] submit_bridge_fdbm_frac_* now runs UniDB. Prefer submit_bridge_unidb_unet_afno_controlxs_pixel512_big.sh" >&2
fi
PARTITION=${PARTITION:-small-g}
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-56}
TIME_LIMIT=${TIME_LIMIT:-3-00:00:00}
RUN_TASKS=${RUN_TASKS:-8}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
BATCH_PER_GPU=${BATCH_PER_GPU:-1}
EPOCHS=${EPOCHS:-400}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-160}
RUN_BRIDGE_PREFLIGHT=${RUN_BRIDGE_PREFLIGHT:-1}
BRIDGE_PREFLIGHT_STRICT=${BRIDGE_PREFLIGHT_STRICT:-0}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_TAG=${RUN_TAG:-bridge_unidb_unet_afno_controlxs_pixel512_${STAMP}}
cd "$PF_ROOT"
ACC=$((GLOBAL_BATCH/(RUN_TASKS*BATCH_PER_GPU)))
JOBID=$(sbatch -A project_462001338 -p "$PARTITION" -N "$NODES" --ntasks-per-node=1 --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$CPUS_PER_TASK" -t "$TIME_LIMIT" -J brgunidb \
  --export=ALL,RUN_TAG="$RUN_TAG",CFG_BASE="$CFG_BASE",MODE=strong,RUN_TASKS="$RUN_TASKS",GLOBAL_BATCH="$GLOBAL_BATCH",BATCH_PER_GPU="$BATCH_PER_GPU",EPOCHS="$EPOCHS",STEPS_PER_EPOCH="$STEPS_PER_EPOCH",USE_VAL=1,DETERMINISTIC_OVERRIDE=0,CLEAR_RESUME=1,RUN_BRIDGE_PREFLIGHT="$RUN_BRIDGE_PREFLIGHT",BRIDGE_PREFLIGHT_STRICT="$BRIDGE_PREFLIGHT_STRICT" \
  "$PF_ROOT/slurm/lumi_scale_aif_torchrun_quick.sh" | awk '{print $4}')
OUT_DIR="$PF_ROOT/runs/${RUN_TAG}_strong_n${NODES}_ws${RUN_TASKS}_bpg${BATCH_PER_GPU}_acc${ACC}_${JOBID}"
mkdir -p "$STACK_ROOT/runs/big"
ln -sfn "$OUT_DIR" "$STACK_ROOT/runs/big/${RUN_TAG}_${JOBID}"
echo "submitted_training_job=$JOBID"
echo "out_dir=$OUT_DIR"
echo "run_link=$STACK_ROOT/runs/big/${RUN_TAG}_${JOBID}"
