#!/bin/bash
#SBATCH --job-name=gputest_unet_bridge_overfit
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export HDF5_USE_FILE_LOCKING=FALSE
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH="${ROOT}"
PY=/scratch/project_2008261/physics_ml/bin/python3.11
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

CFG_BASE=${CFG:-${ROOT}/configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8.yaml}
if [[ ! -f "$CFG_BASE" ]]; then
  echo "Config not found: ${CFG_BASE}" >&2
  exit 2
fi

TMP_DIR=${SLURM_TMPDIR:-/tmp}
TMP_CFG=${TMP_DIR}/cfg_unet_bridge_overfit_${SLURM_JOB_ID:-$$}.yaml

cat <<'PY' > /tmp/build_gputest_overfit_cfg.py
import os
import yaml
import sys

base_cfg = sys.argv[1]
out_cfg = sys.argv[2]
overfit_n = int(os.environ.get('OVERFIT_N', '2'))
overfit_indices_raw = os.environ.get('OVERFIT_INDICES', '').strip()
overfit_indices = [int(tok.strip()) for tok in overfit_indices_raw.split(',') if tok.strip()]
base_out_suffix = os.environ.get('OUT_SUFFIX', 'gputest_overfit')
train_steps = int(os.environ.get('TRAIN_STEPS_PER_EPOCH', '2'))
val_steps = int(os.environ.get('VAL_STEPS_PER_EPOCH', '1'))
seed = int(os.environ.get('SEED', '1'))
force_warmup_zero = os.environ.get('FORCE_WARMUP_ZERO', '1').strip().lower() not in {'0', 'false', 'no'}

with open(base_cfg, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg.setdefault('trainer', {})
cfg.setdefault('loader', {})
cfg.setdefault('dataloader', {})
cfg['trainer']['seed'] = seed
cfg['trainer']['epochs'] = int(os.environ.get('EPOCHS', '1'))
cfg['trainer']['steps_per_epoch'] = train_steps
cfg['trainer']['val_steps_per_epoch'] = val_steps
cfg['trainer']['accumulation_steps'] = 1
cfg['trainer']['use_val'] = True
cfg['loader']['batch_size'] = 1
cfg['loader']['num_workers'] = 0
if overfit_indices:
    cfg['loader']['overfit_indices'] = overfit_indices
    cfg['loader'].pop('overfit_n', None)
else:
    cfg['loader']['overfit_n'] = overfit_n
    cfg['loader'].pop('overfit_indices', None)

cfg['dataloader'].setdefault('train_args', {})
cfg['dataloader'].setdefault('val_args', {})
if overfit_indices:
    # Keep full global indexing when explicit indices are requested.
    cfg['dataloader']['train_args']['limit_per_group'] = None
    cfg['dataloader']['train_args']['max_items'] = None
    cfg['dataloader']['val_args']['limit_per_group'] = None
    cfg['dataloader']['val_args']['max_items'] = None
else:
    cfg['dataloader']['train_args']['limit_per_group'] = 1
    cfg['dataloader']['train_args']['max_items'] = 1
    cfg['dataloader']['val_args']['limit_per_group'] = 1
    cfg['dataloader']['val_args']['max_items'] = 1

# Overfit sanity checks should not inherit long-run LR warmup schedules.
if force_warmup_zero:
    cfg.setdefault('sched', {})
    cfg['sched']['warmup_epochs'] = 0
    cfg['sched']['warmup_start_lr'] = 0.0

out_dir = str(cfg['trainer'].get('out_dir', '')).rstrip('/')
if out_dir:
    cfg['trainer']['out_dir'] = f"{out_dir}_{base_out_suffix}"

with open(out_cfg, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[gputest-overfit] wrote cfg: {out_cfg}")
print(f"[gputest-overfit] out_dir: {cfg['trainer'].get('out_dir')}")
if overfit_indices:
    print(f"[gputest-overfit] overfit_indices: {overfit_indices}")
else:
    print(f"[gputest-overfit] overfit_n: {overfit_n}")
print(f"[gputest-overfit] steps_per_epoch: {train_steps}")
print(f"[gputest-overfit] force_warmup_zero: {force_warmup_zero}")
PY

echo "[gputest-overfit] host=$(hostname) job=${SLURM_JOB_ID:-na}"
echo "[gputest-overfit] base_cfg=${CFG_BASE}"
$PY /tmp/build_gputest_overfit_cfg.py "$CFG_BASE" "$TMP_CFG"

cd "$ROOT"
$PY -m models.train.core.train -c "$TMP_CFG"
