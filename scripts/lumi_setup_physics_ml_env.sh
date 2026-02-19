#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR=${ENV_DIR:-${PROJECT_ROOT}/.venv_physics_ml_lumi}
REQ_FILE=${REQ_FILE:-${PROJECT_ROOT}/env/lumi/requirements-extra.txt}

module purge
module load LUMI
module load partition/G
module load Local-CSC/default
module load pytorch/2.5

BASE_PY=$(which python)
echo "[env-setup] module python: ${BASE_PY}"
"${BASE_PY}" -V

if [[ ! -d "${ENV_DIR}" ]]; then
  echo "[env-setup] creating virtualenv: ${ENV_DIR}"
  "${BASE_PY}" -m venv --system-site-packages "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -f "${REQ_FILE}" ]]; then
  echo "[env-setup] installing extras from ${REQ_FILE}"
  python -m pip install -r "${REQ_FILE}"
fi

echo "[env-setup] sanity check"
python - <<PY
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
mods=["heavyball","pytorch_wavelets","torchcfm","hydra","omegaconf","h5py","mlflow","einops"]
for m in mods:
    try:
        __import__(m)
        print(m, "OK")
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
PY

echo "[env-setup] done"
echo "Activate later with: source ${ENV_DIR}/bin/activate"
