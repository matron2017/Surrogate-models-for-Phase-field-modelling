#!/bin/bash
#SBATCH --job-name=scale_aif_torchrun_quick
#SBATCH --account=project_462001306
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}
# Enforce absolute, stable project root so module imports do not depend on submit cwd.
PROJECT_ROOT=$(cd "${PROJECT_ROOT}" && pwd)
cd "${PROJECT_ROOT}"
CFG_BASE=${CFG_BASE:-${PROJECT_ROOT}/configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b80_rdbmres_afno8_stochastic.yaml}
TRAIN_H5=${TRAIN_H5:-${PROJECT_ROOT}/data/stochastic/rightclean_512lb_touchcut_s8b5/simulation_train_rightclean_fixed34_gradshared_512lb_touchcut_s8b5_v4.h5}
VAL_H5=${VAL_H5:-${PROJECT_ROOT}/data/stochastic/rightclean_512lb_touchcut_s8b5/simulation_val_rightclean_fixed34_gradshared_512lb_touchcut_s8b5_v4.h5}
SIM_MAP=${SIM_MAP:-${PROJECT_ROOT}/data/stochastic/sim_map.json}

AIF_CONTAINER_IMAGE=${AIF_CONTAINER_IMAGE:-/pfs/lustref1/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-torch-u24r64f21m43t29-20260216_093549.sif}
AIF_VENV_DIR=${AIF_VENV_DIR:-${PROJECT_ROOT}/.venv_aif_torch_train}

MODE=${MODE:-strong}
GLOBAL_BATCH=${GLOBAL_BATCH:-128}
BATCH_PER_GPU=${BATCH_PER_GPU:-4}
WEAK_ACCUM_STEPS=${WEAK_ACCUM_STEPS:-1}

EPOCHS=${EPOCHS:-6}
STEPS_MODE=${STEPS_MODE:-optimizer}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-40}
NUM_WORKERS=${NUM_WORKERS:-0}
USE_VAL=${USE_VAL:-0}
RUN_TAG=${RUN_TAG:-scale_aif_torchrun_quick}
PRINT_GPU_SANITY=${PRINT_GPU_SANITY:-1}
ENABLE_COMM_DEBUG=${ENABLE_COMM_DEBUG:-0}
COLLECT_PAIR_STATS=${COLLECT_PAIR_STATS:-0}
DDP_BUCKET_CAP_MB=${DDP_BUCKET_CAP_MB:-128}
DDP_GRAD_BUCKET_VIEW=${DDP_GRAD_BUCKET_VIEW:-1}
DDP_ALLOW_STATIC_GRAPH_WITH_ACCUM=${DDP_ALLOW_STATIC_GRAPH_WITH_ACCUM:-0}
CLEAR_RESUME=${CLEAR_RESUME:-1}
DETERMINISTIC_OVERRIDE=${DETERMINISTIC_OVERRIDE:--1}
RUN_BRIDGE_PREFLIGHT=${RUN_BRIDGE_PREFLIGHT:-0}
BRIDGE_PREFLIGHT_STRICT=${BRIDGE_PREFLIGHT_STRICT:-0}

module --force purge
module load LUMI
module load partition/G
module load Local-CSC/default
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

RUN_TASKS=${RUN_TASKS:-$((SLURM_JOB_NUM_NODES * 8))}
if (( RUN_TASKS < 1 )); then
  echo "RUN_TASKS must be >=1" >&2
  exit 2
fi
if (( RUN_TASKS % SLURM_JOB_NUM_NODES != 0 )); then
  echo "RUN_TASKS=${RUN_TASKS} must divide by nodes=${SLURM_JOB_NUM_NODES}" >&2
  exit 2
fi
NPROC_PER_NODE=$(( RUN_TASKS / SLURM_JOB_NUM_NODES ))
WORLD_SIZE=${RUN_TASKS}

if [[ "${MODE}" == "strong" ]]; then
  DENOM=$(( WORLD_SIZE * BATCH_PER_GPU ))
  if (( GLOBAL_BATCH % DENOM != 0 )); then
    echo "GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by WORLD_SIZE*BATCH_PER_GPU=${DENOM}" >&2
    exit 2
  fi
  ACCUM_STEPS=$(( GLOBAL_BATCH / DENOM ))
else
  ACCUM_STEPS=${WEAK_ACCUM_STEPS}
fi

EFFECTIVE_BATCH=$(( WORLD_SIZE * BATCH_PER_GPU * ACCUM_STEPS ))
if [[ "${STEPS_MODE}" == "optimizer" ]]; then
  OPT_STEPS_PER_EPOCH=${STEPS_PER_EPOCH}
  CFG_STEPS_PER_EPOCH=$(( STEPS_PER_EPOCH * ACCUM_STEPS ))
else
  CFG_STEPS_PER_EPOCH=${STEPS_PER_EPOCH}
  OPT_STEPS_PER_EPOCH=$(( (STEPS_PER_EPOCH + ACCUM_STEPS - 1) / ACCUM_STEPS ))
fi

OUT_DIR=${OUT_DIR:-${PROJECT_ROOT}/runs/${RUN_TAG}_${MODE}_n${SLURM_NNODES}_ws${WORLD_SIZE}_bpg${BATCH_PER_GPU}_acc${ACCUM_STEPS}_${SLURM_JOB_ID}}
CFG_RUN=${PROJECT_ROOT}/tmp/${RUN_TAG}_${MODE}_${SLURM_JOB_ID}.yaml
BRIDGE_PREFLIGHT_JSON=${BRIDGE_PREFLIGHT_JSON:-${OUT_DIR}/bridge_preflight.json}
mkdir -p "${PROJECT_ROOT}/logs/slurm" "${PROJECT_ROOT}/tmp" "${OUT_DIR}"

python3 - <<PY
import yaml
from pathlib import Path
cfg_base = Path("${CFG_BASE}")
cfg_run = Path("${CFG_RUN}")
with cfg_base.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
old_root = "/scratch/project_2008261/pf_surrogate_modelling"
project_root = str(Path("${PROJECT_ROOT}").resolve())

def deep_replace(obj):
    if isinstance(obj, dict):
        return {k: deep_replace(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_replace(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(old_root):
        return obj.replace(old_root, project_root, 1)
    return obj

cfg = deep_replace(cfg)
cfg.setdefault("paths", {}).setdefault("h5", {})
cfg.setdefault("loader", {})
cfg.setdefault("trainer", {})
if "${SIM_MAP}":
    cfg["paths"]["sim_map"] = "${SIM_MAP}"
h5 = cfg["paths"]["h5"]
if "${TRAIN_H5}":
    if isinstance(h5.get("train"), dict):
        h5["train"]["h5_path"] = "${TRAIN_H5}"
    else:
        h5["train"] = "${TRAIN_H5}"
if "${VAL_H5}":
    h5["val"] = "${VAL_H5}"
cfg["trainer"]["out_dir"] = "${OUT_DIR}"
cfg["trainer"]["epochs"] = int(${EPOCHS})
cfg["trainer"]["steps_per_epoch"] = int(${CFG_STEPS_PER_EPOCH})
cfg["trainer"]["accumulation_steps"] = int(${ACCUM_STEPS})
cfg["trainer"]["use_val"] = bool(int(${USE_VAL}))
if "${DETERMINISTIC_OVERRIDE}" in ("0", "1"):
    cfg["trainer"]["deterministic"] = bool(int("${DETERMINISTIC_OVERRIDE}"))
cfg["loader"]["batch_size"] = int(${BATCH_PER_GPU})
cfg["loader"]["num_workers"] = int(${NUM_WORKERS})
diff_cfg = cfg.setdefault("diffusion", {}) if isinstance(cfg, dict) else {}
if isinstance(diff_cfg, dict):
    # Keep bridge validation inference explicit in generated configs.
    diff_cfg.setdefault("val_endpoint_mode", "source_rollout_dbim")
    default_nfe = int(diff_cfg.get("timesteps", 20)) if str(diff_cfg.get("timesteps", "")).strip() else 20
    diff_cfg.setdefault("val_rollout_nfe", default_nfe)
    diff_cfg.setdefault("val_rollout_eta", 0.0)
cfg["trainer"]["collect_pair_stats"] = bool(int(${COLLECT_PAIR_STATS}))
cfg["trainer"]["ddp_gradient_as_bucket_view"] = bool(int(${DDP_GRAD_BUCKET_VIEW}))
cfg["trainer"]["ddp_allow_static_graph_with_accum"] = bool(int(${DDP_ALLOW_STATIC_GRAPH_WITH_ACCUM}))
if int(${CLEAR_RESUME}):
    cfg["trainer"]["resume"] = False
if int(${DDP_BUCKET_CAP_MB}) > 0:
    cfg["trainer"]["ddp_bucket_cap_mb"] = float(${DDP_BUCKET_CAP_MB})
with cfg_run.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[lumi-scale] wrote config: {cfg_run}")
PY

MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)}
MASTER_PORT=${MASTER_PORT:-$((26000 + (SLURM_JOB_ID % 30000)))}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPICH_GPU_SUPPORT_ENABLED=1
export HOME=${HOME_OVERRIDE:-/tmp/${USER:-u}/h${SLURM_JOB_ID}}
export TMPDIR=${TMPDIR_OVERRIDE:-/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME_OVERRIDE:-/tmp/${USER:-u}/c${SLURM_JOB_ID}}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME_OVERRIDE:-/tmp/${USER:-u}/g${SLURM_JOB_ID}}
mkdir -p "${HOME}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}"
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
export FI_PROVIDER=${FI_PROVIDER:-cxi}
export FI_CXI_DISABLE_CQ_HUGETLB=${FI_CXI_DISABLE_CQ_HUGETLB:-1}
export OFI_NCCL_LOG_LEVEL=${OFI_NCCL_LOG_LEVEL:-WARN}
export MIOPEN_NODE_ROOT_BASE=${MIOPEN_NODE_ROOT_BASE:-/tmp/${USER:-u}-miopen-${SLURM_JOB_ID}}
export MIOPEN_DEBUG_DISABLE_FIND_DB=${MIOPEN_DEBUG_DISABLE_FIND_DB:-1}
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export PF_DISABLE_CUDNN_BENCHMARK=${PF_DISABLE_CUDNN_BENCHMARK:-1}
# ws32 on ROCm showed DDP init instability with explicit device_ids; default to implicit mapping.
export PF_DDP_EXPLICIT_DEVICE_IDS=${PF_DDP_EXPLICIT_DEVICE_IDS:-0}
# Optional override for DDP init_sync (empty = framework default).
export PF_DDP_INIT_SYNC=${PF_DDP_INIT_SYNC:-}
if [[ "${ENABLE_COMM_DEBUG}" == "1" ]]; then
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}
  export NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-${OUT_DIR}/nccl_%h_%p.log}
  export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
  export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-INFO}
fi

export PROJECT_ROOT MASTER_ADDR MASTER_PORT NPROC_PER_NODE AIF_CONTAINER_IMAGE AIF_VENV_DIR CFG_RUN

echo "[lumi-scale] mode=${MODE} nodes=${SLURM_NNODES} world_size=${WORLD_SIZE} batch_per_gpu=${BATCH_PER_GPU} accum=${ACCUM_STEPS} effective_batch=${EFFECTIVE_BATCH}"
echo "[lumi-scale] steps_mode=${STEPS_MODE} micro_steps_per_epoch=${CFG_STEPS_PER_EPOCH} optimizer_steps_per_epoch=${OPT_STEPS_PER_EPOCH}"
echo "[lumi-scale] cfg=${CFG_RUN} out_dir=${OUT_DIR}"
echo "[lumi-scale] launch_mode=torchrun_per_node nproc_per_node=${NPROC_PER_NODE}"
echo "[lumi-scale] net_ifname NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
echo "[lumi-scale] ofi FI_PROVIDER=${FI_PROVIDER} FI_CXI_DISABLE_CQ_HUGETLB=${FI_CXI_DISABLE_CQ_HUGETLB} OFI_NCCL_LOG_LEVEL=${OFI_NCCL_LOG_LEVEL}"
echo "[lumi-scale] miopen_node_root_base=${MIOPEN_NODE_ROOT_BASE} miopen_disable_find_db=${MIOPEN_DEBUG_DISABLE_FIND_DB} miopen_find_mode=${MIOPEN_FIND_MODE} cudnn_benchmark_disabled=${PF_DISABLE_CUDNN_BENCHMARK}"
echo "[lumi-scale] ddp_explicit_device_ids=${PF_DDP_EXPLICIT_DEVICE_IDS} ddp_init_sync=${PF_DDP_INIT_SYNC:-default}"
echo "[lumi-scale] trainer_knobs collect_pair_stats=${COLLECT_PAIR_STATS} ddp_bucket_cap_mb=${DDP_BUCKET_CAP_MB} ddp_grad_bucket_view=${DDP_GRAD_BUCKET_VIEW} ddp_allow_static_graph_with_accum=${DDP_ALLOW_STATIC_GRAPH_WITH_ACCUM} clear_resume=${CLEAR_RESUME}"
echo "[lumi-scale] bridge_preflight enabled=${RUN_BRIDGE_PREFLIGHT} strict=${BRIDGE_PREFLIGHT_STRICT} json=${BRIDGE_PREFLIGHT_JSON}"
echo "[lumi-scale] home=${HOME} tmpdir=${TMPDIR} xdg_cache=${XDG_CACHE_HOME}"
echo "[lumi-scale] aif_container_image=${AIF_CONTAINER_IMAGE}"
echo "[lumi-scale] aif_venv_dir=${AIF_VENV_DIR}"

if [[ "${RUN_BRIDGE_PREFLIGHT}" == "1" ]]; then
  preflight_args=(--config "${CFG_RUN}" --json-out "${BRIDGE_PREFLIGHT_JSON}")
  if [[ "${BRIDGE_PREFLIGHT_STRICT}" == "1" ]]; then
    preflight_args+=(--strict)
  fi
  singularity exec --rocm "${AIF_CONTAINER_IMAGE}" "${AIF_VENV_DIR}/bin/python" \
    "${PROJECT_ROOT}/scripts/bridge_preflight_check.py" "${preflight_args[@]}"
fi

if [[ "${PRINT_GPU_SANITY}" == "1" ]]; then
  singularity exec --rocm "${AIF_CONTAINER_IMAGE}" python - <<'PYCHK'
import torch
print("[lumi-scale] torch.version.cuda=%s" % getattr(torch.version, "cuda", None))
print("[lumi-scale] torch.version.hip=%s" % getattr(torch.version, "hip", None))
print("[lumi-scale] cuda_available=%s device_count=%s" % (torch.cuda.is_available(), torch.cuda.device_count()))
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("[lumi-scale] device0=%s" % torch.cuda.get_device_name(0))
PYCHK
fi

run_t0=$(date +%s)
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpu-bind=cores bash -c '
  set -euo pipefail
  local_node_id=${SLURM_NODEID:-0}
  local_miopen_root="${MIOPEN_NODE_ROOT_BASE}-n${local_node_id}"
  export MIOPEN_USER_DB_PATH="${local_miopen_root}/userdb"
  export MIOPEN_CUSTOM_CACHE_DIR="${local_miopen_root}/cache"
  mkdir -p "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"
  mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" || true
  export SINGULARITYENV_HOME="${HOME}"
  export SINGULARITYENV_TMPDIR="${TMPDIR}"
  export SINGULARITYENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"
  export SINGULARITYENV_XDG_CONFIG_HOME="${XDG_CONFIG_HOME}"
  export SINGULARITYENV_MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH}"
  export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR}"
  export SINGULARITYENV_MIOPEN_DEBUG_DISABLE_FIND_DB="${MIOPEN_DEBUG_DISABLE_FIND_DB}"
  export SINGULARITYENV_MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE}"
  export SINGULARITYENV_PF_DISABLE_CUDNN_BENCHMARK="${PF_DISABLE_CUDNN_BENCHMARK}"
  export SINGULARITYENV_PF_DDP_EXPLICIT_DEVICE_IDS="${PF_DDP_EXPLICIT_DEVICE_IDS}"
  export SINGULARITYENV_PF_DDP_INIT_SYNC="${PF_DDP_INIT_SYNC}"
  export SINGULARITYENV_FI_PROVIDER="${FI_PROVIDER}"
  export SINGULARITYENV_FI_CXI_DISABLE_CQ_HUGETLB="${FI_CXI_DISABLE_CQ_HUGETLB}"
  export SINGULARITYENV_OFI_NCCL_LOG_LEVEL="${OFI_NCCL_LOG_LEVEL}"
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
  cd "${PROJECT_ROOT}"
  exec singularity exec --rocm "${AIF_CONTAINER_IMAGE}" "${AIF_VENV_DIR}/bin/python" -m torch.distributed.run \
    --nnodes="${SLURM_JOB_NUM_NODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m models.train.core.train -c "${CFG_RUN}"
'
run_rc=$?
run_t1=$(date +%s)
echo "[lumi-scale] run_rc=${run_rc} elapsed_seconds=$((run_t1 - run_t0))"
exit ${run_rc}
