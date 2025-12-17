# Rapid Solidification — Remaining TODOs

Only active work items are tracked here (completed tasks removed for brevity).

## 1. Config propagation in auxiliary launchers
- [ ] Verify any non-standard launchers (Optuna scripts, bespoke notebooks) set the descriptor fields before invoking `models/train/core/train.py`.

## 3. Registries stay authoritative
- [ ] When adding new backbones or adaptive hooks, extend the registries and avoid gating logic inside `models/train/core/train.py`.

## 5. Optuna + Slurm integration
- [ ] Ensure the Optuna driver writes sampled hyperparameters into the shared config schema prior to submission.
- [ ] Update worker scripts so each trial runs `models/train/core/train.py` with the merged config and reports `metric.val_rel_l2` consistently to both Optuna and MLflow.

## Diffusion Follow-ups
- [ ] Run the residual DDPM pipeline on the real HDF5 dataset (swap out the placeholder path in `configs/ddpm_placeholder.json`).
- [ ] Expand the inference CLI to tile residual predictions back into full-resolution fields and export diagnostics.

## Queue + Wavelet follow-ups
- [ ] Watch UNet smoke `30910647` and U-AFNO smoke `30910648`; resubmit U-AFNO or FNO smoke once queue limits clear.
- [ ] After smokes land, review `logs/slurm/training_*30910647*.out` / `training_*30910648*.out` for sane loss trends and errors.
- [ ] Confirm wavelet YAMLs point to `/scratch/project_2008261/solidification_modelling/data/rapid_solidification/simulation_train_a10000b50000.wavelet.h5` before longer runs.

## Boundary conditions (all backbones)
- [ ] Enforce mixed BCs consistently: y periodic (circular padding), x-left replicate (copy edge outward), x-right reflect/mirror (≈ zero normal gradient). For patch training, use halo + center loss: crop (P+2H)×(P+2H), pad with `pad_mixed_bc`, predict, compute loss on center P×P only. Re-impose BCs after each diffusion/flow step if needed (wrap y, replicate left, reflect right).
