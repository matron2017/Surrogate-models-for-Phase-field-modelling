# Next Steps (AE PSGD Debugging)

## Current Objective
Stabilise strict‑sync PSGD training at scale and eliminate early NaNs while keeping baseline runs untouched for comparison.

## Current State (at time of writing)
- Baseline full runs (unchanged):
  - 31422326 (nowavelet, 5 nodes)
  - 31422330 (wavelet, 5 nodes)
- Strict‑sync diagnostic run:
  - 31459094 (nowavelet strict‑sync, 5 nodes, AMP **off**) — RUNNING

## What we know
- Short strict‑sync runs (120–350 steps, 2–6 nodes) complete without NaNs.
- Full strict‑sync runs with **baseline hyperparams** (batch=2, AMP on) fail **immediately at epoch 1 step 3** across many ranks.
- Warmup + preconditioner ramp did **not** prevent step‑3 NaNs.

## Immediate Decision Tree
1) **If 31459094 (AMP off) still hits step‑3 NaNs**
   - AMP is **not** the primary cause.
   - Next actions:
     - Enable per‑layer forward stats (min/max/mean) around step 1–5.
     - Log loss components and check for invalid operations (e.g., division/log/sqrt) in loss path.
     - Consider temporarily disabling saturate/softclip or lowering `saturate_bound` to test stability.

2) **If 31459094 is stable (no step‑3 NaNs)**
   - AMP is likely the trigger.
   - Next actions:
     - Re‑enable AMP but set GradScaler `init_scale` lower (if configurable) or switch to static loss scaling.
     - Consider `bf16` only if GPUs support it (V100 does not). Otherwise, stay AMP off for strict‑sync while keeping baselines as AMP on.

## Other hypotheses still open
- Data‑order sensitivity: strict‑sync runs at world size 20/24 expose early “bad batches.”
  - If AMP off also fails, add a short “bad batch dump” (gid/pair) for the first NaN.
- PSGD preconditioner instability: only relevant if NaNs occur **later** (epoch 2+). Not supported by current evidence.

## Where we track progress
- Debug journal: `pf_surrogate_modelling/docs/AE_PSGD_DEBUG_LOG.md`
- Slurm logs: `pf_surrogate_modelling/logs/slurm/`
- Launchers: `pf_surrogate_modelling/slurm/`
- Configs: `pf_surrogate_modelling/configs/train/`

## Notes
- Keep baseline runs (31422326/31422330) untouched for apples‑to‑apples comparison.
- Strict‑sync is diagnostic: if it stays unstable, the issue is likely numerical/AMP/data‑order rather than NCCL desync.

## Related
- See `OPEN_QUESTIONS.md` for thermal-field conditioning decisions and open research questions.
