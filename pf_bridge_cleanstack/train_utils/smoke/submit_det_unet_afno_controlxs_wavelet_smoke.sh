#!/usr/bin/env bash
set -euo pipefail

STACK_ROOT=/scratch/project_462001338/pf_bridge_cleanstack
PF_ROOT=/scratch/project_462001338/pf_surrogate_modelling
CFG_BASE=${CFG_BASE:-$STACK_ROOT/configs/pf_surrogates/smoke/train_det_unet_afno_controlxs_wavelet_512_smoke.yaml}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_TAG=${RUN_TAG:-smoke_det_unet_afno_controlxs_wavelet_512_${STAMP}}

cd "$PF_ROOT"
JOBID=$(sbatch -A project_462001338 -p dev-g -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=7 -t 00:10:00 -J smk_det_afno \
  --export=ALL,RUN_TAG="$RUN_TAG",CFG_BASE="$CFG_BASE",MODE=strong,RUN_TASKS=1,GLOBAL_BATCH=1,BATCH_PER_GPU=1,EPOCHS=1,STEPS_PER_EPOCH=10,USE_VAL=1,DETERMINISTIC_OVERRIDE=0,CLEAR_RESUME=1 \
  "$PF_ROOT/slurm/lumi_scale_aif_torchrun_quick.sh" | awk "{print \$4}")

CFG_RUN="$PF_ROOT/tmp/${RUN_TAG}_strong_${JOBID}.yaml"
for _ in $(seq 1 80); do
  ST=$(squeue -h -j "$JOBID" -o %T || true)
  [[ -z "$ST" ]] && break
  sleep 10
done

OUT_DIR=$(python3 - <<PY
import yaml
print(yaml.safe_load(open("$CFG_RUN"))["trainer"]["out_dir"])
PY
)

mkdir -p "$STACK_ROOT/runs/smoke"
ln -sfn "$OUT_DIR" "$STACK_ROOT/runs/smoke/${RUN_TAG}_${JOBID}"

echo "submitted_training_job=$JOBID"
echo "config_run=$CFG_RUN"
echo "out_dir=$OUT_DIR"
echo "smoke_link=$STACK_ROOT/runs/smoke/${RUN_TAG}_${JOBID}"
