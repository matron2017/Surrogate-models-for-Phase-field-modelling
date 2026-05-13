#!/bin/bash
# Resume det_px_full training from checkpoint (job 34391106 reached 24h limit ~epoch 138)
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

PY=/scratch/project_2008261/physics_ml/bin/python

# Reuse the existing run directory from job 34391106
OUT_DIR=$DET/runs/big_det_unet_afno_controlxs_wavelet_512_20260511T113243Z_n3_ws12_bpg1_acc6_34391106
CKPT=$OUT_DIR/UNetFiLMAttn/checkpoint.last.pth
CFG_SNAPSHOT=$OUT_DIR/UNetFiLMAttn/config_snapshot.yaml
TMP_CFG=$DET/tmp/det_resume_${SLURM_JOB_ID}.yaml

NODES=${SLURM_NNODES:-3}
GPUS_PER_NODE=4
WORLD_SIZE=$((NODES * GPUS_PER_NODE))

mkdir -p "$DET/tmp" "$DET/logs/slurm"

# Copy config snapshot as runtime config (has correct out_dir, epochs=260, etc.)
cp "$CFG_SNAPSHOT" "$TMP_CFG"

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29544}

echo "[det-pixel-resume] cfg=$TMP_CFG"
echo "[det-pixel-resume] out_dir=$OUT_DIR"
echo "[det-pixel-resume] ckpt=$CKPT"
echo "[det-pixel-resume] nodes=$NODES master=${MASTER_HOST}:${MASTER_PORT}"
nvidia-smi || true

cd "$ROOT"
srun --ntasks="$NODES" --ntasks-per-node=1 \
  "$PY" -m torch.distributed.run \
    --nnodes="${NODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_endpoint="${MASTER_HOST}:${MASTER_PORT}" \
    -m models.train.core.train -c "${TMP_CFG}" --resume "${CKPT}"
