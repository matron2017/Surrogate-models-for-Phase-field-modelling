#!/usr/bin/env bash
set -euo pipefail
ROOT=/scratch/project_462001338/pf_bridge_cleanstack
PF=/scratch/project_462001338/pf_surrogate_modelling
CFG=$ROOT/configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_eightpair.yaml
RUN_BRIDGE_PREFLIGHT=${RUN_BRIDGE_PREFLIGHT:-1}
BRIDGE_PREFLIGHT_STRICT=${BRIDGE_PREFLIGHT_STRICT:-0}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_TAG=smoke_bridge_unidb_t25_eightpair_${STAMP}
AIF_IMG=/pfs/lustref1/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-torch-u24r64f21m43t29-20260216_093549.sif
AIF_PY="$PF/.venv_aif_torch_train/bin/python"

cd "$PF"
JOBID=$(sbatch -A project_462001338 -p dev-g -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=7 -t 00:25:00 -J ov8brg \
  --export=ALL,RUN_TAG="$RUN_TAG",CFG_BASE="$CFG",MODE=strong,RUN_TASKS=1,GLOBAL_BATCH=1,BATCH_PER_GPU=1,EPOCHS=1,STEPS_PER_EPOCH=260,USE_VAL=1,DETERMINISTIC_OVERRIDE=0,CLEAR_RESUME=1,RUN_BRIDGE_PREFLIGHT="$RUN_BRIDGE_PREFLIGHT",BRIDGE_PREFLIGHT_STRICT="$BRIDGE_PREFLIGHT_STRICT" \
  "$PF/slurm/lumi_scale_aif_torchrun_quick.sh" | awk '{print $4}')

echo "submitted_training_job=$JOBID"
for _ in $(seq 1 180); do
  ST=$(squeue -h -j "$JOBID" -o %T || true)
  [[ -z "$ST" ]] && break
  sleep 10
done

CFG_RUN="$PF/tmp/${RUN_TAG}_strong_${JOBID}.yaml"
OUT_DIR=$(python3 - <<PY
import yaml
print(yaml.safe_load(open("$CFG_RUN"))["trainer"]["out_dir"])
PY
)
CKPT="$OUT_DIR/UNetFiLMAttn/checkpoint.best.pth"
PRE_JSON="$OUT_DIR/bridge_preflight.json"
EVAL_JSON="$OUT_DIR/overfit_eightpair_eval_suite.json"
EVAL_DIR="$OUT_DIR/overfit_eightpair_eval_panels"

if [[ ! -f "$CKPT" ]]; then
  echo "missing_checkpoint=$CKPT" >&2
  exit 2
fi

EWRAP=$(cat <<EOF
set -euo pipefail
module --force purge >/dev/null 2>&1
module load LUMI partition/G Local-CSC/default >/dev/null 2>&1
module use /appl/local/laifs/modules >/dev/null 2>&1
module load lumi-aif-singularity-bindings >/dev/null 2>&1
  singularity exec --rocm "$AIF_IMG" "$AIF_PY" "$ROOT/scripts/bridge_preflight_check.py" --config "$CFG_RUN" --json-out "$PRE_JSON"
singularity exec --rocm "$AIF_IMG" "$AIF_PY" "$ROOT/train_utils/smoke/eval_overfit_unidb_suite.py" --ckpt "$CKPT" --split val --nfe 25 --max-items 8 --device cuda --seed 1 --out-json "$EVAL_JSON" --out-dir "$EVAL_DIR"
EOF
)

EJOB=$(sbatch -A project_462001338 -p dev-g -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=7 -t 00:20:00 -J ev8brg \
  --output "$PF/logs/slurm/ev8brg_%j.out" --wrap "$EWRAP" | awk '{print $4}')

echo "submitted_eval_job=$EJOB"
for _ in $(seq 1 240); do
  ST=$(squeue -h -j "$EJOB" -o %T || true)
  [[ -z "$ST" ]] && break
  sleep 5
done

if [[ -f "$EVAL_JSON" ]]; then
  python3 - <<PY
import json
p="$EVAL_JSON"
d=json.load(open(p))
print("eval_aggregate="+json.dumps(d.get("aggregate", {}), sort_keys=True))
PY
else
  echo "missing_eval_json=$EVAL_JSON" >&2
fi

echo "run_tag=$RUN_TAG"
echo "config_run=$CFG_RUN"
echo "out_dir=$OUT_DIR"
echo "checkpoint=$CKPT"
echo "preflight_json=$PRE_JSON"
echo "eval_json=$EVAL_JSON"
echo "eval_panels=$EVAL_DIR"
