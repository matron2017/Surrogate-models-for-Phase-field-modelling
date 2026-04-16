# Deterministic Surrogate Evidence (2026-04-16)

This note records the recovered evidence that the **non-diffusion deterministic surrogate** path worked for the PF next-step task.

## What model this is

Canonical overfit config:
- `/scratch/project_462001338/pf_bridge_cleanstack/configs/pf_surrogates/smoke/train_det_unet_afno_controlxs_wavelet_512_overfit40.yaml`

Relevant config facts:
- `train.model_family: surrogate`
- Predicts next state from current state, not a diffusion objective
- `dataloader.args.input_channels: [0, 1]`
- `dataloader.args.target_channels: [0, 1]`
- `dataloader.args.add_thermal: true`
- `conditioning.use_theta: true`
- `model.params.use_time: false`
- `model.params.use_control_branch: true`

Interpretation:
- Input is the current PF state `x_t` (channels 0/1) plus thermal conditioning
- Target is the next PF state `x_{t+1}` (channels 0/1)
- This is the direct deterministic next-step surrogate path

## Recovered overfit40 evidence

Primary run:
- Run dir: `/scratch/project_462001338/pf_surrogate_modelling/runs/overfit40_det_20260415T124759Z_strong_n1_ws1_bpg1_acc1_17540045/UNetFiLMAttn`
- Best checkpoint: `/scratch/project_462001338/pf_surrogate_modelling/runs/overfit40_det_20260415T124759Z_strong_n1_ws1_bpg1_acc1_17540045/UNetFiLMAttn/checkpoint.best.pth`
- Run metadata: `/scratch/project_462001338/pf_surrogate_modelling/runs/overfit40_det_20260415T124759Z_strong_n1_ws1_bpg1_acc1_17540045/UNetFiLMAttn/run.json`
- Metrics: `/scratch/project_462001338/pf_surrogate_modelling/runs/overfit40_det_20260415T124759Z_strong_n1_ws1_bpg1_acc1_17540045/UNetFiLMAttn/metrics.csv`

Recovered run status from `run.json`:
- `status.state: completed`
- `status.final_epoch: 8`
- `status.best_metric: 0.00022676473436149535`
- Monitor split/mode: `val`, `min`

Recovered validation metrics from `metrics.csv` (`split=val` rows):

| Metric | First val | Best val | Best epoch | Last val |
| --- | ---: | ---: | ---: | ---: |
| objective | 0.01720768 | 0.00022676 | 7 | 0.00072488 |
| mae | 0.07547544 | 0.00630994 | 7 | 0.01276333 |
| vrmse | 0.31511925 | 0.03379518 | 7 | 0.05709305 |

The overfit run therefore improved strongly from epoch 1 to epoch 7 and finished with a still-low validation objective at epoch 8.

## Recovered full-run evidence

Longer deterministic run:
- Run dir: `/scratch/project_462001338/pf_surrogate_modelling/runs/big_det_unet_afno_controlxs_wavelet_512_20260415T110328Z_strong_n1_ws8_bpg1_acc1_17537001/UNetFiLMAttn`
- Best checkpoint: `/scratch/project_462001338/pf_surrogate_modelling/runs/big_det_unet_afno_controlxs_wavelet_512_20260415T110328Z_strong_n1_ws8_bpg1_acc1_17537001/UNetFiLMAttn/checkpoint.best.pth`
- Metrics: `/scratch/project_462001338/pf_surrogate_modelling/runs/big_det_unet_afno_controlxs_wavelet_512_20260415T110328Z_strong_n1_ws8_bpg1_acc1_17537001/UNetFiLMAttn/metrics.csv`

Recovered validation summary (`split=val` rows):
- Best `objective`: `0.00211436` at epoch `13`
- Best `mae`: `0.01287281` at epoch `9`
- Best `vrmse`: `0.05952924` at epoch `13`

This gives a second independent preserved run showing the deterministic path trained successfully beyond the 40-pair overfit check.

## Preserved visual evidence

Deterministic comparison plots exist in:
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_idx0.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_idx39.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_pair001.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_pair220.png`
- `/scratch/project_462001338/pf_bridge_cleanstack/visuals/smoke/overfit40_20260415T124759Z/det_pair260.png`

These files were present and non-empty when this note was written.

## Conclusion

The deterministic surrogate path has preserved evidence of working:
- correct model family (`surrogate`)
- correct conditioning path (thermal/theta enabled, no diffusion time input)
- completed overfit and longer runs with checkpoints
- strong recovered validation metrics
- preserved deterministic comparison plots

What is still **not** re-verified from this shell:
- a brand-new rerun on LUMI in this session
- remote inspection through the old `ssh lumi` alias, which is not configured in the current shell environment

So the historical result is recovered and documented, but a fresh rerun remains optional future confirmation rather than part of this evidence note.
