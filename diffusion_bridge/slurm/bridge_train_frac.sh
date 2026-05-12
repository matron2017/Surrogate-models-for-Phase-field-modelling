#!/bin/bash
# Fractional bridge full training — 3 nodes × 4 GPUs = 12 ranks, 300M+ backbone
#SBATCH --job-name=bridge_frac
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/diffusion_bridge/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/diffusion_bridge/logs/slurm/%x_%j.err

set -euo pipefail

ROOT=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates
VENV_DIR=${VENV_DIR:-/scratch/project_2008261/solidification_modelling/physics_ml}
[ -x "${VENV_DIR}/bin/python" ] || VENV_DIR=/scratch/project_2008261/physics_ml
PY="${VENV_DIR}/bin/python"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-${ROOT}/diffusion_bridge/configs/frac_bridge_pde_512_big.yaml}
NODES=${SLURM_NNODES:-3}
GPUS_PER_NODE=4

MASTER_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk 'NR==1{print $1}')
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR=$("$PY" - "${MASTER_HOST}" <<'PY'
import socket, sys
host = sys.argv[1]
try: print(socket.gethostbyname(host))
except Exception: print("")
PY
)
fi
[[ -z "${MASTER_ADDR}" ]] && MASTER_ADDR="${MASTER_HOST}"
MASTER_PORT=${MASTER_PORT:-29546}
export MASTER_ADDR MASTER_PORT

mkdir -p "${ROOT}/diffusion_bridge/logs/slurm"

echo "======================================================="
echo " FracBridge FULL  job=${SLURM_JOB_ID}"
echo " nodes=${NODES}  gpus/node=${GPUS_PER_NODE}  world=$((NODES * GPUS_PER_NODE))"
echo " master=${MASTER_ADDR}:${MASTER_PORT}"
echo " config=${CONFIG}"
echo "======================================================="
nvidia-smi || true

srun --ntasks="${NODES}" --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  node_rank=${SLURM_PROCID}
  '"$PY"' -m torch.distributed.run \
    --nnodes='"$NODES"' \
    --nproc_per_node='"$GPUS_PER_NODE"' \
    --node_rank=${node_rank} \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    '"$ROOT"'/diffusion_bridge/scripts/train_bridge.py \
      --config '"$CONFIG"'
'

echo "FracBridge training DONE"
