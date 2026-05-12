#!/bin/bash
#SBATCH --job-name=det_px_ddpchk
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel/logs/slurm/%x_%j.err
#SBATCH --chdir=/scratch/project_2008261/pf_surrogate_modelling/Phase_field_surrogates/deterministic_pixel

set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export PYTHONUNBUFFERED=1

ROOT=/scratch/project_2008261/pf_surrogate_modelling
export PYTHONPATH=$ROOT:${PYTHONPATH:-}

PY=/scratch/project_2008261/solidification_modelling/physics_ml/bin/python
if [[ ! -x "$PY" ]]; then
  PY=/scratch/project_2008261/physics_ml/bin/python
fi
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

NODES=${SLURM_NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
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
MASTER_PORT=${MASTER_PORT:-29542}
export MASTER_ADDR MASTER_PORT

echo "[det-pixel-ddp-check] nodes=$NODES gpus_per_node=$GPUS_PER_NODE master_host=$MASTER_HOST master_addr=$MASTER_ADDR master_port=$MASTER_PORT"
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
    -m models.train.core.ddp_multi_node_check
'
