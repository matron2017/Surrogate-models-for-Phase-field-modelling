# PF Bridge Clean Stack

Clean operational stack for PF surrogate and bridge experiments on LUMI project `462001338`.

## Current runtime roots
- Training code root: `/scratch/project_462001338/pf_surrogate_modelling`
- Cleanstack orchestration root: `/scratch/project_462001338/pf_bridge_cleanstack`
- Deterministic evidence note: `docs/DETERMINISTIC_SURROGATE_EVIDENCE_2026-04-16.md`
- LUMI workflow note: `docs/LUMI_BATCH_AND_PYTHON_WORKFLOW_2026-04-16.md`
- Measurement/show note: `docs/PF_SURROGATE_MEASURE_AND_SHOW_2026-04-16.md`

## Canonical bridge family now
- Active bridge development is **pure UniDB** in this stack.
- Fractional/FDBM references remain under `external_refs/` and historical docs, but are not the default active path.

## Canonical bridge entries
- Big config: `configs/pf_surrogates/big/train_bridge_unidb_unet_afno_controlxs_pixel512_big.yaml`
- Big submit: `train_utils/big/submit_bridge_unidb_unet_afno_controlxs_pixel512_big.sh`
- Smoke one-pair: `configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_onepair.yaml`
- Smoke tiny-set: `configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_eightpair.yaml`
- Preflight: `scripts/bridge_preflight_check.py`
- Cleanstack bridge launchers now enable bridge preflight automatically and write `bridge_preflight.json` into the run `trainer.out_dir`.

## Legacy naming note
Some legacy file names include `fdbm_frac` / `bridge_frac` for backward compatibility. They are wrappers and now point to UniDB configs with warning messages. Prefer the explicit `*_unidb_*` entries.

## Dataset links used by cleanstack
- `/scratch/project_462001338/pf_bridge_cleanstack/data/train.h5`
- `/scratch/project_462001338/pf_bridge_cleanstack/data/val.h5`
- `/scratch/project_462001338/pf_bridge_cleanstack/data/test.h5`
