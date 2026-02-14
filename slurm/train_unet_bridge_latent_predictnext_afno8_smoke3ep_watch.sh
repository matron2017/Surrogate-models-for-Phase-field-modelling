#!/bin/bash
#SBATCH --job-name=latent_unet_bridge_afno8_smoke3ep_watch
#SBATCH --account=project_2008261
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/project_2008261/pf_surrogate_modelling/logs/slurm/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export CUBLAS_WORKSPACE_CONFIG=:16:8
export GIT_PYTHON_REFRESH=quiet
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ib0}
export GLOO_SOCKET_FAMILY=${GLOO_SOCKET_FAMILY:-AF_INET}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT}

ROOT=/scratch/project_2008261/pf_surrogate_modelling
PY=/scratch/project_2008261/physics_ml/bin/python3.11
CFG=${CFG:-${ROOT}/configs/train/train_diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8_smoke3ep.yaml}
AE_CKPT=${AE_CKPT:-/scratch/project_2008261/solidification_modelling/runs/ae_latent_lola_big_64_1024_psgd_uncached_freq1_12g_latent32_nowavelet_rightclean_fixed34_gradshared_b40_precond64_p128/LatentAELoLAModel/checkpoint.best.pth}
INDICES=${INDICES:-234,255,300}
BRIDGE_NFE=${BRIDGE_NFE:-20}
PLOT_POLL_SECONDS=${PLOT_POLL_SECONDS:-45}
PLOT_DEVICE=${PLOT_DEVICE:-cpu}

RUN_BASE=/scratch/project_2008261/pf_surrogate_modelling/runs/diffusion_bridge_unet_thermal_latentpsgd_e279_gpu12h_1n4g_b64_rdbmres_predictnext_nomass_afno8_smoke3ep
RUN_DIR=${RUN_BASE}/UNetFiLMAttn
PLOT_OUT=${PLOT_OUT:-${ROOT}/results/visuals/bridge_afno8_smoke3ep_epoch_watch_${SLURM_JOB_ID}}

if [[ ! -x "${PY}" ]]; then
  PY=python3
fi
if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}" >&2
  exit 2
fi
if [[ ! -f "${AE_CKPT}" ]]; then
  echo "AE checkpoint not found: ${AE_CKPT}" >&2
  exit 2
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}"
nvidia-smi || true

echo "[run] training config: ${CFG}"
echo "[run] checkpoint watch dir: ${RUN_DIR}"
echo "[run] plot out: ${PLOT_OUT}"

${PY} -m torch.distributed.run --nproc_per_node=4 -m models.train.core.train -c "${CFG}" &
TRAIN_PID=$!

cleanup() {
  if [[ -n "${WATCH_PID:-}" ]] && kill -0 "${WATCH_PID}" 2>/dev/null; then
    kill "${WATCH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

${PY} "${ROOT}/scripts/watch_bridge_checkpoint_and_plot.py" \
  --run-dir "${RUN_DIR}" \
  --ae-ckpt "${AE_CKPT}" \
  --out-dir "${PLOT_OUT}" \
  --indices "${INDICES}" \
  --bridge-nfe "${BRIDGE_NFE}" \
  --poll-seconds "${PLOT_POLL_SECONDS}" \
  --train-pid "${TRAIN_PID}" \
  --python "${PY}" \
  --device "${PLOT_DEVICE}" &
WATCH_PID=$!

wait "${TRAIN_PID}"
TRAIN_RC=$?
wait "${WATCH_PID}" || true

exit "${TRAIN_RC}"
