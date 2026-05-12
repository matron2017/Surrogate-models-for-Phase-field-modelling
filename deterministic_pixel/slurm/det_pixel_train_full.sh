#!/bin/bash
#SBATCH --job-name=det_px_full
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel

set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export HDF5_USE_FILE_LOCKING=FALSE
export GIT_PYTHON_REFRESH=quiet
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1

ROOT=/scratch/project_2008261/pf_surrogate_modelling
PHASE=$ROOT/Phase_field_surrogates
DET=$PHASE/deterministic_pixel
export PYTHONPATH=$ROOT:${PYTHONPATH:-}

PY=/scratch/project_2008261/solidification_modelling/physics_ml/bin/python
if [[ ! -x "$PY" ]]; then
  PY=/scratch/project_2008261/physics_ml/bin/python
fi
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

CFG_BASE=${CFG:-$DET/configs_current/pf_surrogates/big/train_det_unet_afno_controlxs_wavelet_512_big.yaml}
NODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
WORLD_SIZE=$((NODES * GPUS_PER_NODE))
BATCH_PER_RANK=${BATCH_PER_RANK:-1}
GLOBAL_BATCH=${GLOBAL_BATCH:-72}   # 3 nodes x 4 GPUs = 12 ranks; nearest clean multiple >=64 is 72 (accum=6)
if (( GLOBAL_BATCH % (WORLD_SIZE * BATCH_PER_RANK) != 0 )); then
  echo "GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by WORLD_SIZE*BATCH_PER_RANK=$((WORLD_SIZE * BATCH_PER_RANK))" >&2
  exit 2
fi
ACCUM_STEPS=$(( GLOBAL_BATCH / (WORLD_SIZE * BATCH_PER_RANK) ))
TRAIN_EPOCHS=${TRAIN_EPOCHS:-260}
TRAIN_STEPS_PER_EPOCH=${TRAIN_STEPS_PER_EPOCH:-0}
TRAIN_USE_VAL=${TRAIN_USE_VAL:-1}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-4}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_TAG=${RUN_TAG:-big_det_unet_afno_controlxs_wavelet_512_${STAMP}}
OUT_DIR=${OUT_DIR:-$DET/runs/${RUN_TAG}_n${NODES}_ws${WORLD_SIZE}_bpg${BATCH_PER_RANK}_acc${ACCUM_STEPS}_${SLURM_JOB_ID}}
TMP_CFG=${TMP_CFG:-$DET/tmp/${RUN_TAG}_${SLURM_JOB_ID}.yaml}

mkdir -p "$DET/tmp" "$DET/logs/slurm" "$OUT_DIR"

"$PY" "$DET/scripts/build_det_pixel_runtime_cfg.py" \
  --base-config "$CFG_BASE" \
  --out-config "$TMP_CFG" \
  --out-dir "$OUT_DIR" \
  --epochs "$TRAIN_EPOCHS" \
  --steps-per-epoch "$TRAIN_STEPS_PER_EPOCH" \
  --batch-per-rank "$BATCH_PER_RANK" \
  --accumulation-steps "$ACCUM_STEPS" \
  --num-workers "$TRAIN_NUM_WORKERS" \
  --use-val "$TRAIN_USE_VAL"

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1{print $1}')
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR=$("$PY" - "${MASTER_HOST}" <<'PY'
import socket, sys
host = sys.argv[1]
try:
    print(socket.gethostbyname(host))
except Exception:
    print("")
PY
)
fi
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="${MASTER_HOST}"
fi
MASTER_PORT=${MASTER_PORT:-29544}
export MASTER_ADDR MASTER_PORT

echo "[det-pixel-full] cfg=$TMP_CFG"
echo "[det-pixel-full] out_dir=$OUT_DIR"
echo "[det-pixel-full] world_size=$WORLD_SIZE batch_per_rank=$BATCH_PER_RANK accum=$ACCUM_STEPS effective_batch=$((WORLD_SIZE * BATCH_PER_RANK * ACCUM_STEPS))"
echo "[det-pixel-full] master_host=$MASTER_HOST master_addr=$MASTER_ADDR master_port=$MASTER_PORT"
nvidia-smi || true

cd "$ROOT"
srun --ntasks="$NODES" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  node_rank=${SLURM_PROCID}
  '"$PY"' -m torch.distributed.run \
    --nnodes='"$NODES"' \
    --nproc_per_node='"$GPUS_PER_NODE"' \
    --node_rank=${node_rank} \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    -m models.train.core.train -c '"$TMP_CFG"'
'
