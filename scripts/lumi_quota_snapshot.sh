#!/bin/bash
set -euo pipefail

ACCOUNT=${ACCOUNT:-project_462001306}
SCRATCH_PATH=${SCRATCH_PATH:-/scratch/${ACCOUNT}}

echo "== Host =="
hostname

echo
echo "== Account associations (Slurm) =="
sacctmgr show assoc where user=$USER account=${ACCOUNT} format=cluster,account,partition,maxjobs,maxsubmitjobs,maxwall -n || true

echo
echo "== Partition limits (LUMI GPU) =="
for p in dev-g small-g standard-g; do
  scontrol show partition ${p} | grep -E "PartitionName|MaxNodes|MaxTime" || true
  echo
 done

echo "== Scratch quota (group) =="
lfs quota -h -g ${ACCOUNT} ${SCRATCH_PATH} 2>/dev/null || true

echo "== Scratch quota (user) =="
lfs quota -h -u $USER ${SCRATCH_PATH} 2>/dev/null || true

echo "== Scratch usage snapshot =="
du -sh ${SCRATCH_PATH}/pf_surrogate_modelling 2>/dev/null || true
