# LUMI vs Puhti: Quotas, Limits, and Development Use (2026-02-19)

## 1. Current LUMI Project Resources (from your project panel)

Project `462001306`:
- GPU hours: `5000h`
- CPU core-hours: `500000h`
- Storage-hours: `22000 TiBh`

These are the principal budget constraints for benchmark planning.

## 2. LUMI Scheduling Limits (observed via Slurm)

Account: `project_462001306`

Partition limits:
- `dev-g`: max `32` nodes, max walltime `03:00:00`
- `small-g`: max `4` nodes, max walltime `3-00:00:00`
- `standard-g`: max `1024` nodes, max walltime `2-00:00:00`

Association limits (selected):
- `dev-g`: max jobs `2`, max submit jobs `2`
- `small-g`: max jobs `200`, max submit jobs `210`
- `standard-g`: max jobs `200`, max submit jobs `210`

## 3. Puhti Scheduling Limits (observed for project_2008261)

Partition limits:
- `gputest`: max `2` nodes, max walltime `00:15:00`
- `gpu`: max `20` nodes, max walltime `3-00:00:00`
- `interactive`: max `1` node, max walltime `7-00:00:00`

Association limits (selected):
- `gputest`: max jobs `1`, max submit jobs `2`
- `gpu`: max jobs `200`, max submit jobs `400`

## 4. Storage Snapshot Comparison (current user usage)

From Lustre quota snapshots:
- LUMI `/scratch/project_462001306`: user usage about `6.2G`
- Puhti `/scratch/project_2008261`: user usage about `738.7G`

Note: `lfs quota` on both systems shows default block/inode quota fields; the
project-level allocation constraints for GPU/CPU/storage-hours should be taken
from CSC project allocation reporting.

## 5. What to Use for Development vs Benchmarking

Recommended split:

1. Puhti (development and rapid iteration)
- Existing CUDA-oriented environment and legacy scripts already mature.
- Good for short model/debug cycles before LUMI transfer.

2. LUMI `dev-g` (portability check)
- fast smoke tests, environment sanity, script validation.

3. LUMI `small-g` (structured benchmark matrix up to 4 nodes)
- strong and weak scaling baseline runs.

4. LUMI `standard-g` (extreme-scale evidence)
- >4 node evidence points and production-like pilot runs.

## 6. Practical Resource Envelope with 5000 GPUh

Given full-node usage at 8 GPU units/node:
- `5000 GPUh` = `625 node-hours`

This is enough for:
- complete smoke campaign,
- strong + weak scaling matrices with repeats,
- several longer multi-node pilot runs,

provided runs are planned and measured carefully.

## 7. Commands

Refresh LUMI limits/quotas quickly:
- `scripts/lumi_quota_snapshot.sh`

Run matrix submission (dry-run first):
- `cd slurm`
- `DRY_RUN=1 ./lumi_g_submit_backbone_scaling_matrix.sh`

## 8. References

- LUMI access modes and policies: https://research.csc.fi/lumi-access
- CSC extremely large resources calls: https://research.csc.fi/resources/applying-for-resources/extremely-large-resources/
- LUMI partition docs: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/partitions/
