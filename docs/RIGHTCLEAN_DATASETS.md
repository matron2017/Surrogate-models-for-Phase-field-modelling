# Right-Edge Cleaned Datasets (fixed34, grad-shared)

Date: 2026-02-06

## Summary
We generate "rightclean" datasets by truncating each simulation time series
after a shared cutoff per thermal gradient. The cutoff is defined as the
earliest seed (within the same thermal gradient) that touches a fixed cut line.

This ensures all seeds for a given gradient have the same last time step.

## Fixed cut line
- Cut line width: 34 columns from the right edge.
- The cut line position was chosen to match:
  `results/right_buffer_diagnostics/test/last_frames_gradbuf10/G1.6e6_seed1_sim_0051_last_frame.png`

## Touch definition
For each sim/seed, we find the first time step where the rightmost 34 columns
are NOT all identical to the last column. This is the first frame where the
dendrite reaches the fixed cut line (x = W - 34).

## Shared cutoff per thermal gradient
For each thermal gradient, we take the earliest touch across seeds and use that
as the shared cutoff for all seeds in that gradient.

This is recorded in:
- `results/right_buffer_diagnostics/{split}/touch_frames_fixed34_gradshared.csv`
- `results/right_buffer_diagnostics/{split}/grad_cutoffs_touch_fixed34.csv`

## Output datasets (created by job 31653488)
Written under:
`data/stochastic/rightclean/`

- `simulation_train_rightclean_fixed34_gradshared.h5`
- `simulation_val_rightclean_fixed34_gradshared.h5`
- `simulation_test_rightclean_fixed34_gradshared.h5`

Cutoff is inclusive (cutoff_frame is the last kept frame).

Per-split truncation summaries:
- `results/right_buffer_diagnostics/{split}/truncate_fixed34_gradshared_summary.csv`

## Plots (for verification)
Fixed cut line (same for all):
- `results/right_buffer_diagnostics/{split}/last_frames_fixed34_touch/`

Shared cutoff per gradient:
- `results/right_buffer_diagnostics/{split}/last_frames_fixed34_gradshared_touch/`

## Scripts
- Touch detection (fixed line): `scripts/right_buffer_touch_by_cutline.py`
- Share earliest touch per gradient: `scripts/right_buffer_touch_grad_shared.py`
- Truncate H5 by cutoff CSV: `scripts/right_buffer_truncate_by_cutoff_csv.py`
