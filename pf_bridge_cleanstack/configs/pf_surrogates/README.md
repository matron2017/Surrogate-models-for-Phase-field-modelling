# PF Surrogate Configs

## Canonical active configs (UniDB)
- `big/train_bridge_unidb_unet_afno_controlxs_pixel512_big.yaml`
- `smoke/train_smoke_bridge_unidb_t25_onepair.yaml`
- `smoke/train_smoke_bridge_unidb_t25_eightpair.yaml`
- `smoke/train_bridge_unidb_unet_afno_controlxs_pixel512_overfit40.yaml`

## Strict separation rule
Do not mix UniDB and fractional/FDBM keys in one config.
- UniDB: `noise_schedule: unidb_*` + `diffusion_objective: unidb_*|predict_next`
- Fractional/FDBM: `noise_schedule: bridge_*` + `diffusion_objective: fdbm_drift_*`

Preflight check:
- `python scripts/bridge_preflight_check.py --config <config.yaml>`
- Smoke launchers in `train_utils/smoke/` use the same cleanstack entrypoint.
- Cleanstack bridge launchers also run this automatically before training and store `bridge_preflight.json` under the run output directory.
