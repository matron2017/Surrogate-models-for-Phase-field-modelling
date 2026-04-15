# FDBM Loss Port Notes (Paired Pixel Bridge)

## Goal
Use an FDBM-style drift target in our PyTorch training stack (not UniDB).

## Reference formula (FDBM_paired)
From `external_refs/FDBM_paired/fdbm/training/losses.py`:
- `drift_true = (pos_T - pos_t) / beta_t_diff`
- then MSE/L2 between predicted drift and `drift_true`.

## Our objective
- Objective key: `fdbm_drift_mse`
- Implemented in: `pf_surrogate_modelling/models/train/loss_registry.py`

For bridge schedules:
- `x_t` corresponds to `noisy`
- `x_T` corresponds to `target`
- `beta_t_diff` is mapped to `(sigma^2) * (1 - a_t)`
  where `a_t` is the target-endpoint coefficient from bridge schedule
  (`x_t = a_t * x_T + b_t * x_0 + c_t * eps`).

This keeps the FDBM drift form while using our bridge coefficient parameterization.

## Fractional bridge schedule
- Schedule key: `bridge_fractional`
- Kind: `bridge` (distinct from UniDB)
- Implemented in: `pf_surrogate_modelling/models/diffusion/scheduler_registry.py`

## Cleanstack big-run entry points
- Config:
  `configs/pf_surrogates/big/train_bridge_fdbm_frac_unet_afno_controlxs_pixel512_big.yaml`
- Submit script:
  `train_utils/big/submit_bridge_fdbm_frac_unet_afno_controlxs_pixel512_big.sh`

## Current active submission
- Job: `17536613`
- Partition: `small-g`
- Tag prefix: `bridge_fdbm_frac_unet_afno_controlxs_pixel512_...`
