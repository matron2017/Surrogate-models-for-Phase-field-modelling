# GPU Monitoring & Diagnostics (Puhti/Mahti)

This cheat sheet distils CSC’s scattered documentation into a single place so every GPU job (single-node or multi-node) can be checked quickly. All commands assume Puhti, but the same tools exist on Mahti.

---

## 1. After the Job: `seff` + `sacct`

Always start with the Slurm accounting tools—they summarise CPU, memory, GPU load and CSC billing units.

```bash
# Overall efficiency summary (CPU, memory, GPU)
seff <jobid>

# Detailed accounting rows for every step in the job
sacct -j <jobid> \
      -o jobid,jobname,partition,allocnodes,alloccpus,elapsed,state,reqtres%40
```

Interpretation tips:

- `CPU Efficiency` < 100 % means many reserved cores were idle; adjust `--cpus-per-task` or reduce worker threads.
- `Memory Efficiency` shows how close you were to the requested memory. If it stays <10 %, you can lower `--mem` next time to reduce queue pressure.
- GPU jobs display per-device load/memory/energy. If the `GPU load` table reports ≈0 % while the job is “COMPLETED”, you likely never called CUDA (or the workload was too short to register); double-check your launcher.

Example (`seff 30878234`):

```
CPU Efficiency: 3.55% of 03:16:00 core-walltime
GPU load:
  r01g01 gpu0-3 → 0%; r02g01 gpu0-3 → 9–15% (short test run)
```

---

## 2. Live Monitoring (during a run)

Inside an interactive allocation or via `srun --pty`:

```bash
# One-off snapshot
nvidia-smi

# Continuous view (1 s interval) with the useful columns
watch -n 1 nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv
```

For multi-node jobs, run on every node simultaneously:

```bash
srun -N "$SLURM_NNODES" -n "$SLURM_NNODES" bash -c '
  hostname
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
'
```

This confirms that each GPU is actually being used and helps spot imbalanced workloads.

---

## 3. Lightweight Logger Hook (optional)

Add this snippet to the top of long-running scripts to log utilisation once per minute without babysitting the job:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 60 > gpuutil_${SLURM_JOB_ID}.csv &
GPU_LOG_PID=$!

# ... your srun/python command ...

kill $GPU_LOG_PID 2>/dev/null || true
```

The CSV can be inspected post-run or attached to troubleshooting reports.

---

## 4. Scaling & Parallelisation Checks

When increasing nodes/GPUs, measure whether wall-clock time improves proportionally.

1. Run short “scaling” cases with identical inputs but varying `--nodes × --gres=gpu`.
2. Record `sacct` elapsed times and `seff` GPU loads for each run.
3. Stop scaling up when wall-time reductions are marginal or GPU load collapses.

Template command to gather a concise table:

```bash
sacct -S 2025-12-09 -u "$USER" \
  -o jobid,jobname,allocnodes,reqtres%30,elapsed,state \
  | grep rs_ddp_2x4
```

---

## 5. Deep Profiling (only when needed)

If `seff`/`nvidia-smi` reveal low utilisation but the reason is unclear, use the profilers installed on CSC systems:

- **Nsight Systems (`nsys`)** – timeline of CPU↔GPU activity, NCCL waits, data transfers.
- **Nsight Compute (`ncu`)** – kernel-level occupancy, memory throughput, tensor-core usage.
- **Scalasca / Score-P / VTune** – for CPU/MPI-heavy sections.

Example (Nsight Systems over an `srun`):

```bash
module load nvidia-profile
nsys profile -o nsys_report_${SLURM_JOB_ID} \
  srun python -m models.train.core.train -c configs/train/...yaml
```

Inspect the `.qdrep` in Nsight Systems locally to see where GPUs idle.

---

## 6. Multi-Node Validation Checklist

Use `slurm/test_parallel_multinode.sh` with `train/core/ddp_multi_node_check.py` to ensure cross-node NCCL works before queueing long runs.

What to look for in the job log:

- Host mapping lines show two different nodes (`r01g01`, `r02g01`, …) and all ranks report `all_reduce sum=28.0`.
- The training step prints two GPU tables: the pre-run baseline (util ≈0 %) and the in-run snapshot (util >80 % when the workload is healthy).
- The appended `seff`/`sacct` summaries confirm `AllocNodes=2` and list GPU loads per host.

If any rank stays stuck on one hostname, re-queue on the `gpu` partition and verify project quota; if NCCL reports `Address family not supported by protocol`, force IPv4 via `export NCCL_SOCKET_IFNAME=ib0` or adjust `MASTER_ADDR` to the IPv4 address of the first node.

---

Keep this reference handy whenever you need to justify resource usage, debug scaling regressions, or hand over diagnostics to CSC support. The commands mirror their official guidelines but are tuned to the models workflow.***
