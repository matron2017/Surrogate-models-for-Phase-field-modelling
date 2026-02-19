# Repo Reorg Plan (Safe With Pending Jobs)

Date: 2026-02-17  
Scope: simplify repository structure without changing behavior of pending jobs.

## Safety First (Do Not Touch While Jobs Are Pending)

Pending jobs currently queued:

- `31891745` (`latent_unet_flowmatch24h_rdbm_afno8_b64_1n4g_sto`)
- `31891763` (`latent_unet_sfm24h_rdbm_afno8_b64_1n4g`)

Do not modify these paths until those jobs have started+snapshotted config or finished:

- `slurm/train_unet_flowmatch_latent_stochastic_24h_1n4g_b64.sh`
- `slurm/train_unet_flowmatch_latent_sfm_24h_1n4g_b64.sh`
- `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_stochastic.yaml`
- `configs/train/train_flowmatch_unet_thermal_latentpsgd_e279_gpu24h_1n4g_b64_rdbmres_afno8_sfm_latent_long_v1.yaml`
- `models/train/core/train.py`
- `models/train/core/loops.py`
- `models/train/loss_registry.py`
- `models/diffusion/scheduler_registry.py`

Reason: SLURM jobs read files at runtime in this workspace.

## Current Friction

- Too many flat files in `configs/train/`, `slurm/`, and `scripts/`.
- Mixed lifecycle in one folder (active run configs + old smoke/ablation configs).
- Similar functionality split across multiple script namespaces (`scripts/`, `visuals/`, `tools/`).
- Large generated outputs in `results/` and `logs/slurm/` make navigation noisy.

## Non-Breaking Target Layout

Keep runtime behavior stable while improving discoverability:

```text
configs/
  train/
    active/        # current supported training configs
    ablation/      # controlled ablations
    smoke/         # gputest/smoke configs
    archive/       # frozen historical configs
slurm/
  train/
  eval/
  diag/
  smoke/
scripts/
  eval/
  diag/
  plot/
  data/
docs/
  workflows/
  reports/
  reference/
```

## Migration Strategy

### Phase 0 (Now, while jobs pending)

- Documentation only.
- Create explicit freeze list (above).
- Inventory active entrypoints and mark ownership.

### Phase 1 (After pending jobs finish)

- Move files to grouped folders.
- Keep old paths as thin wrappers for one transition cycle:
  - old SLURM path -> wrapper that calls new path.
  - old script path -> wrapper importing/running new module.
- Verify wrappers by running one `gputest` smoke per active training family.

### Phase 2

- Remove wrappers after one stable cycle.
- Move old inactive configs to `archive/`.
- Add index docs:
  - `configs/train/INDEX.md` (active canonical configs only)
  - `slurm/INDEX.md` (exact launchers by purpose)
  - `scripts/INDEX.md` (eval/diag/plot entrypoints)

### Phase 3

- Add CI-style lint checks for repo hygiene:
  - no new flat config files outside grouped folders
  - no new diagnostics without a docs link
  - no launcher without canonical config reference

## Practical Rules For ML Research + Engineering

- Every long-run launcher must pin one canonical config.
- Every config used for long runs must include:
  - explicit `train/val` paths
  - monitor metric
  - stochastic eval policy (if stochastic objective)
- Diagnostics must compare against a copy baseline where relevant.
- Keep generated artefacts out of source-tree navigation paths in docs.

## Suggested First Concrete Cleanup Batch (post-pending jobs)

1. Group `slurm/` into `train`, `eval`, `diag`, `smoke` with compatibility wrappers.
2. Group `configs/train/` into `active`, `ablation`, `smoke`, `archive`.
3. Move plotting scripts into `scripts/plot/` and keep old-path wrappers.
4. Add concise `INDEX.md` files for each grouped folder.

