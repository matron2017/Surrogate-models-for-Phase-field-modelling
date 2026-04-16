# PF Surrogate: What to Measure and Show (2026-04-16)

This note answers two separate questions:

1. What plot evidence already exists for the deterministic and UniDB PF surrogate paths?
2. What should be measured and shown next?

## 1. Input/output semantics

### Deterministic surrogate

Config:
- `/scratch/project_462001338/pf_bridge_cleanstack/configs/pf_surrogates/smoke/train_det_unet_afno_controlxs_wavelet_512_overfit40.yaml`

Semantics:
- PF state input: `2 x 512 x 512`
- Thermal conditioning input: `1 x 512 x 512`
- PF target output: `2 x 512 x 512`

Implementation details:
- `dataloader.args.input_channels: [0, 1]`
- `dataloader.args.target_channels: [0, 1]`
- `dataloader.args.add_thermal: true`
- `conditioning.use_theta: true`
- `model.params.in_channels: 2`
- `model.params.hint_channels: 1`
- `model.params.out_channels: 2`

So yes: this is the direct **2 field in + 1 thermal condition + 2 field out** deterministic path.

### UniDB bridge surrogate

Config:
- `/scratch/project_462001338/pf_bridge_cleanstack/configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_onepair.yaml`

Raw data semantics:
- PF source state: `2 x 512 x 512`
- Thermal conditioning: `1 x 512 x 512`
- PF target state: `2 x 512 x 512`

Implementation details:
- `dataloader.args.input_channels: [0, 1]`
- `dataloader.args.target_channels: [0, 1]`
- `dataloader.args.add_thermal: true`
- `conditioning.use_theta: true`
- `model.params.out_channels: 2`
- `model.params.hint_channels: 1`
- `diffusion.schedule_kwargs.input_mode: delta_source_concat`

Important note:
- the **physical task** is still `2 + thermal -> 2`
- but the UniDB model tensor uses `in_channels: 4` because the schedule builds a `delta_source_concat` representation internally

## 2. Existing plot evidence

### Deterministic plots

Confirmed existing:
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_idx0.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_idx39.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_pair001.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_pair220.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_pair260.png`

These come from the deterministic surrogate path and are already good evidence that the plotting pipeline exists for the direct next-step model.

### UniDB plots

Confirmed existing UniDB-linked artifacts:
- `/scratch/project_462001338/pf_surrogate_modelling/runs/smoke_bridge_unidb_t25_onepair_20260416T080126Z_strong_n1_ws1_bpg1_acc1_17558217/overfit_onepair_eval_panels/sample_000_idx_00000_sim_0041_pair_00000.png`
- `/scratch/project_462001338/pf_surrogate_modelling/runs/smoke_bridge_unidb_t25_onepair_20260416T071440Z_strong_n1_ws1_bpg1_acc1_17557068/overfit_onepair_plot.png`
- `/scratch/project_462001338/pf_surrogate_modelling/runs/smoke_bridge_unidb_t25_onepair_20260416T071440Z_strong_n1_ws1_bpg1_acc1_17557068/overfit_onepair_plot_rerun.png`
- `/scratch/project_462001338/pf_surrogate_modelling/runs/smoke_bridge_unidb_t25_eightpair_20260416T073845Z_strong_n1_ws1_bpg1_acc1_17557478/overfit_eightpair_plot.png`

So yes, there are **actual UniDB smoke/eval plots** tied to UniDB checkpoints.

## 3. Important distinction: did this session create them?

No new deterministic or UniDB plots were generated in this session.

What this session did:
- recovered and checked that the deterministic plots exist
- recovered and checked that UniDB smoke/eval plots exist
- tied the UniDB artifacts to actual `smoke_bridge_unidb_*` run folders instead of the older non-UniDB bridge plots
- documented the evidence in cleanstack docs

## 4. What to measure and show now

### A. Deterministic model: the current best “show me it works” package

Use the deterministic path as the main positive baseline.

Show:
1. Aggregate validation metrics:
   - `objective`
   - `mae`
   - `vrmse`
   - `psnr`
2. Copy-baseline comparison:
   - `rmse_pred`
   - `rmse_copy`
   - `ratio_pred_over_copy`
3. Qualitative panels for fixed pairs:
   - source
   - target
   - prediction
   - residual (`pred - gt`)

Current strongest deterministic numbers already recovered:
- overfit40 best val `objective`: `0.00022676`
- best val `mae`: `0.00630994`
- best val `vrmse`: `0.03379518`

Best existing visual examples:
- `det_pair001.png`
- `det_pair220.png`
- `det_pair260.png`

### B. UniDB model: the current best “show me whether it is useful” package

For UniDB, the immediate question is not beauty but **does it beat the copy baseline and does thermal matter?**

Show:
1. Endpoint RMSE vs copy baseline:
   - `val_endpoint_rmse`
   - `copy_baseline_rmse`
   - `copy_gap`
   - `beats_copy`
2. Channel-wise endpoint error:
   - `phi_rmse`
   - `c_rmse`
3. Conditioning-ablation metrics:
   - normal theta
   - shuffled theta
   - zero theta
4. Fixed eval panels from `eval_overfit_unidb_suite.py`

Current recovered one-pair UniDB result is **bad**, not good:
- `val_endpoint_rmse`: `2.482047`
- `copy_baseline_rmse`: `0.308329`
- `copy_gap`: `-2.173718`
- `beats_copy: false`

That means the current one-pair UniDB smoke model is not yet a convincing PF surrogate result.

## 5. Recommended presentation order

If you need to present results now, do it in this order:

1. **Deterministic baseline**
   - lead with metrics + `det_pair001/220/260.png`
   - show it is a functioning PF next-step surrogate with thermal conditioning
2. **UniDB diagnostic status**
   - show the one-pair/eight-pair panels
   - explicitly report copy-baseline failure
   - frame UniDB as still under debugging, not yet competitive

## 6. Practical next step

If the goal is to convince ourselves or others that the PF surrogate setup works, the deterministic model should be the current reference result.

If the goal is to improve UniDB, the next required gate is simple:
- make UniDB beat the copy baseline on one-pair and then on eight-pair
- only after that does broader qualitative plotting become meaningful
