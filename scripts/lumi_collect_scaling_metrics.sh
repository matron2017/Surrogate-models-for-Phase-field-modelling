#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <jobid> [logs_dir]" >&2
  exit 2
fi

JOBID=$1
LOGS_DIR=${2:-logs/slurm}

echo "== sacct summary =="
sacct -j "${JOBID}" --format=JobID,JobName,Partition,AllocTRES,Elapsed,ElapsedRaw,TotalCPU,MaxRSS,State,ExitCode -P

echo
if command -v sstat >/dev/null 2>&1; then
  echo "== sstat batch step =="
  sstat -j "${JOBID}.batch" --format=AveCPU,AveRSS,MaxRSS,MaxVMSize || true
fi

echo
LOG_FILE=$(ls -1t "${LOGS_DIR}"/*"_${JOBID}.out" 2>/dev/null | head -n 1 || true)
if [[ -n "${LOG_FILE}" ]]; then
  echo "== log file =="
  echo "${LOG_FILE}"
  echo "== throughput/time hints =="
  grep -E "(step|samples|throughput|sec/step|epoch|loss)" "${LOG_FILE}" | tail -n 120 || true
else
  echo "No matching log file found in ${LOGS_DIR} for job ${JOBID}."
fi
