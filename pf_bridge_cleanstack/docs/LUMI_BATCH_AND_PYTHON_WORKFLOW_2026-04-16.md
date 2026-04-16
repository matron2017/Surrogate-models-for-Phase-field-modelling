# LUMI Batch and Python Workflow (2026-04-16)

This note records the **current working LUMI launch pattern** for the cleanstack so the correct runtime is not lost again.

## Short answer

Yes: the intended path is clear from the checked-in launchers.

For this stack, the correct execution model is:
- submit through the cleanstack wrappers in `/scratch/project_462001338/pf_bridge_cleanstack/train_utils/...`
- let them call the shared LUMI launcher in `/scratch/project_462001338/pf_surrogate_modelling/slurm/lumi_scale_aif_torchrun_quick.sh`
- run Python inside the AI Factory Singularity container with the project venv at `PF_ROOT/.venv_aif_torch_train`

Do **not** rely on the bare login-shell `python3` for runtime checks. In this shell it is too old for current training utilities.

## Canonical runtime roots

- Cleanstack root: `/scratch/project_462001338/pf_bridge_cleanstack`
- Training root: `/scratch/project_462001338/pf_surrogate_modelling`
- Shared launcher: `/scratch/project_462001338/pf_surrogate_modelling/slurm/lumi_scale_aif_torchrun_quick.sh`

## Correct LUMI module and container environment

The shared launcher uses this module sequence:
- `module --force purge`
- `module load LUMI`
- `module load partition/G`
- `module load Local-CSC/default`
- `module use /appl/local/laifs/modules`
- `module load lumi-aif-singularity-bindings`

The canonical container image is:
- `/pfs/lustref1/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-torch-u24r64f21m43t29-20260216_093549.sif`

The intended Python executable is:
- `${PROJECT_ROOT}/.venv_aif_torch_train/bin/python`

Inside the shared launcher this becomes:
- `singularity exec --rocm "${AIF_CONTAINER_IMAGE}" "${AIF_VENV_DIR}/bin/python" -m torch.distributed.run ...`

## What to avoid

1. Do not run training or preflight checks with the bare system `python3` if you have not entered the container/venv path.
2. Do not launch long production jobs directly from legacy roots when a cleanstack wrapper exists.
3. Do not mix UniDB keys with fractional/FDBM keys in one config.

## Queue policy

Use:
- `dev-g` for smoke/debug/short checks
- `small-g` for longer or production-like training

Current cleanstack wrappers already follow this split:
- deterministic smoke: `train_utils/smoke/submit_det_unet_afno_controlxs_wavelet_smoke.sh` uses `dev-g`
- bridge smoke: `train_utils/smoke/run_overfit_unidb_onepair_dev_g.sh` and `run_overfit_unidb_eightpair_dev_g.sh` use `dev-g`
- canonical big UniDB bridge: `train_utils/big/submit_bridge_unidb_unet_afno_controlxs_pixel512_big.sh` defaults to `small-g`

## Canonical entrypoints

### Deterministic next-step surrogate

- Smoke submit: `/scratch/project_462001338/pf_bridge_cleanstack/train_utils/smoke/submit_det_unet_afno_controlxs_wavelet_smoke.sh`
- Big submit: `/scratch/project_462001338/pf_bridge_cleanstack/train_utils/big/submit_det_unet_afno_controlxs_wavelet_big.sh`

This is the non-diffusion path:
- `train.model_family: surrogate`
- no diffusion time input
- direct next-state prediction with thermal conditioning

Recovered evidence for that path is recorded in:
- `/scratch/project_462001338/pf_bridge_cleanstack/docs/DETERMINISTIC_SURROGATE_EVIDENCE_2026-04-16.md`

### Canonical UniDB bridge

- Big config: `/scratch/project_462001338/pf_bridge_cleanstack/configs/pf_surrogates/big/train_bridge_unidb_unet_afno_controlxs_pixel512_big.yaml`
- Big submit: `/scratch/project_462001338/pf_bridge_cleanstack/train_utils/big/submit_bridge_unidb_unet_afno_controlxs_pixel512_big.sh`
- Smoke one-pair: `/scratch/project_462001338/pf_bridge_cleanstack/configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_onepair.yaml`
- Smoke eight-pair: `/scratch/project_462001338/pf_bridge_cleanstack/configs/pf_surrogates/smoke/train_smoke_bridge_unidb_t25_eightpair.yaml`

## Generated run files

The shared launcher writes:
- generated config: `${PROJECT_ROOT}/tmp/${RUN_TAG}_${MODE}_${SLURM_JOB_ID}.yaml`
- output dir: `${PROJECT_ROOT}/runs/${RUN_TAG}_${MODE}_n${SLURM_NNODES}_ws${WORLD_SIZE}_bpg${BATCH_PER_GPU}_acc${ACCUM_STEPS}_${SLURM_JOB_ID}`

Cleanstack wrappers then create convenience symlinks under:
- `/scratch/project_462001338/pf_bridge_cleanstack/runs/smoke`
- `/scratch/project_462001338/pf_bridge_cleanstack/runs/big`

## Bridge preflight behavior

The cleanstack bridge submitters now pass:
- `RUN_BRIDGE_PREFLIGHT=1`
- `BRIDGE_PREFLIGHT_STRICT=0` by default

That causes the shared launcher to run:
- `/scratch/project_462001338/pf_surrogate_modelling/scripts/bridge_preflight_check.py`

before training starts, with JSON written to:
- `${OUT_DIR}/bridge_preflight.json`

This is only enabled by the bridge wrappers; it is not forced globally for unrelated training jobs.

## Safe mental model

If you want the correct environment, think:

1. **Pick the cleanstack wrapper**
2. **Let it submit to LUMI**
3. **Let the shared launcher build the config**
4. **Let Singularity + `.venv_aif_torch_train` provide Python**

If any check is done outside that path, treat it as potentially misleading until you confirm the same command inside the container/venv runtime.

## Main current risks

The main things to worry about are limited and concrete:
- local shell aliases like `ssh lumi` may not exist everywhere
- bare login-shell Python may be too old for current utilities
- UniDB/FDBM config mixing is still an easy failure mode if bypassing the cleanstack wrappers

Those are workflow risks, not evidence that the cleanstack launch pattern itself is wrong.
