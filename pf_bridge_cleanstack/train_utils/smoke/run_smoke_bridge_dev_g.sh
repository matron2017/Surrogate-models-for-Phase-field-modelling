#!/usr/bin/env bash
set -euo pipefail
ROOT=/scratch/project_462001338/pf_bridge_cleanstack
PF=/scratch/project_462001338/pf_surrogate_modelling
CFG=$ROOT/configs/pf_surrogates/smoke/train_smoke_bridge_unet_afno_controlxs_predictnext_512.yaml
RUN_BRIDGE_PREFLIGHT=${RUN_BRIDGE_PREFLIGHT:-1}
BRIDGE_PREFLIGHT_STRICT=${BRIDGE_PREFLIGHT_STRICT:-0}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_TAG=smoke_bridge_unet_afno_controlxs_predictnext_512_${STAMP}

cd "$PF"
JOBID=$(sbatch -A project_462001338 -p dev-g -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=7 -t 00:10:00 -J smkbrgafno \
  --export=ALL,RUN_TAG="$RUN_TAG",CFG_BASE="$CFG",MODE=strong,RUN_TASKS=1,GLOBAL_BATCH=1,BATCH_PER_GPU=1,EPOCHS=1,STEPS_PER_EPOCH=10,USE_VAL=1,DETERMINISTIC_OVERRIDE=0,CLEAR_RESUME=1,RUN_BRIDGE_PREFLIGHT="$RUN_BRIDGE_PREFLIGHT",BRIDGE_PREFLIGHT_STRICT="$BRIDGE_PREFLIGHT_STRICT" \
  "$PF/slurm/lumi_scale_aif_torchrun_quick.sh" | awk '{print $4}')

echo "submitted_training_job=$JOBID"
for _ in $(seq 1 84); do
  ST=$(squeue -h -j "$JOBID" -o %T || true)
  if [[ -z "$ST" ]]; then
    break
  fi
  sleep 10
done

CFG_RUN="$PF/tmp/${RUN_TAG}_strong_${JOBID}.yaml"
OUT_DIR=$(python3 - <<PY
import yaml
print(yaml.safe_load(open('$CFG_RUN'))['trainer']['out_dir'])
PY
)
CKPT="$OUT_DIR/UNetFiLMAttn/checkpoint.best.pth"
PLOT="$OUT_DIR/smoke_plot.png"
if [[ ! -f "$CKPT" ]]; then
  echo "missing_checkpoint=$CKPT" >&2
  exit 2
fi
module --force purge >/dev/null 2>&1
module load LUMI partition/G Local-CSC/default >/dev/null 2>&1
module use /appl/local/laifs/modules >/dev/null 2>&1
module load lumi-aif-singularity-bindings >/dev/null 2>&1
singularity exec --rocm /pfs/lustref1/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-torch-u24r64f21m43t29-20260216_093549.sif "$PF/.venv_aif_torch_train/bin/python" "$ROOT/train_utils/smoke/smoke_plot_bridge_pixel.py" --ckpt "$CKPT" --index 0 --split val --nfe 25 --out "$PLOT" --device cpu

echo "run_tag=$RUN_TAG"
echo "config_run=$CFG_RUN"
echo "out_dir=$OUT_DIR"
echo "checkpoint=$CKPT"
echo "plot=$PLOT"
