# Workspace Repo Collection

Last updated: **2026-02-17**
Workspace root: `/scratch/project_2008261`

Purpose: one place to track all nearby repos/snapshots and how they relate to the PF latent PDE forecasting work.

## A) Core PF Project

| Local path | Remote | Branch@HEAD | Primary role | Stochastic model path |
|---|---|---|---|---|
| `pf_surrogate_modelling` | `git@github.com:matron2017/Surrogate-models-for-Phase-field-modelling.git` | `main@9cb0097` | Main training/eval codebase for latent AE + diffusion bridge + flow matching PDE surrogate | Yes (diffusion bridge, stochastic flow/SFM configs) |

## B) Direct Method Repos (Flow / Diffusion / Bridge / PDE)

| Local path | Remote / source | Branch@HEAD | Main method framing | Stochastic by design | Notes for PF use |
|---|---|---|---|---|---|
| `PBFM` | `https://github.com/tum-pbs/PBFM` | `main@5a5d13b` | Physics-based flow matching | Yes | Strong reference for physics residual losses with FM |
| `dsbm-pytorch` | `https://github.com/yuyang-shi/dsbm-pytorch.git` | `main@0f3ca75` | Diffusion Schrodinger Bridge Matching (IPF/IMF style) | Yes (bridge SDE modes), RF mode can be deterministic | Good for bridge-time training/sampling logic |
| `UniDB` | `https://github.com/UniDB-SOC/UniDB.git` | `main@5625e73` | Unified diffusion bridge via stochastic optimal control | Yes | Good reference for reverse-step bridge objectives/schedules |
| `RDBM` | `https://github.com/MiliLab/RDBM.git` | `main@ddc8ef9` | Residual diffusion bridge for restoration | Yes | Useful for residual-modulated bridge intuition |
| `DiffusionBridge` | `https://github.com/thu-ml/DiffusionBridge` | `main@9252273` | Diffusion bridge image-to-image models (DBIM/CDBM) | Yes | Useful for solver/sampling baselines |
| `SPL_OFM` | `https://github.com/yzshi5/SPL_OFM.git` | `main@4df44bb` | Operator Flow Matching for stochastic process learning | Yes | Useful for process-level stochastic metrics/ideas |
| `enma` | `https://github.com/itsakk/enma.git` | `main@6956493` | Autoregressive neural PDE operator | Can be stochastic depending on setup | Useful baseline contrast vs FM/bridge approaches |
| `pcfm_upstream` | `https://github.com/cpfpengfei/PCFM` | `main@6bfe94f` | Physics-Constrained Flow Matching | Typically stochastic sampling | Useful for constraint-at-sampling ideas |
| `lola_upstream` | `https://github.com/PolymathicAI/lola` | `main@bd4bdf2` | Latent diffusion for physics emulation | Yes | Strong latent-physics diffusion reference |
| `rectified-flow-pytorch` | `https://github.com/lucidrains/rectified-flow-pytorch` | `main@1a5f58b` | Rectified flow implementations | Usually deterministic ODE, can sample stochastically by variant | Good compact RF implementation reference |

## C) Non-git Snapshots / Aggregated Folders (Still Useful)

| Local path | Source status | Main contents | Stochastic relevance | Notes |
|---|---|---|---|---|
| `DBFM-3E8E` | Snapshot/extracted folder (no `.git` metadata) | Combined flow matching + diffusion bridge scripts | Yes | Keep as historical sandbox reference |
| `diffusion_bridge_unified_framework` | Local assembled framework (no `.git` metadata) | Side-by-side flow matching + diffusion bridge code/notes | Yes | Good comparison directory for quick ablations |

## D) Supporting Generative/Conditioning Libraries

| Local path | Remote | Branch@HEAD | Relevance |
|---|---|---|---|
| `diffusers` | `https://github.com/huggingface/diffusers` | `main@b712042da` | General diffusion tooling and reference implementations |
| `diffusers_clean` | `https://github.com/huggingface/diffusers` | `main@b712042da` | Duplicate clean mirror of diffusers snapshot |
| `ControlNet-XS` | `https://github.com/vislearn/ControlNet-XS.git` | `main@8d90ed4` | Conditional control patterns for diffusion backbones |
| `LoRAdapter` | `https://github.com/CompVis/LoRAdapter.git` | `main@3aa7316` | Conditional adapter design patterns |

## E) Workflow / Infra Repos Nearby

| Local path | Remote | Branch@HEAD | Role |
|---|---|---|---|
| `ML_workflow` | `git@vttgit.vtt.fi:mttmik/tdxdsurrogate.git` | `main@f76c72b` | Puhti workflow, menu tooling, surrogate project orchestration |
| `solidification_modelling` | (contains workflow/data stack) | n/a | Data + run artefact ecosystem used by PF project |
| `alloy_optimization` | `git@dump.vtt.fi:tptatu/alloy_optimization.git` | `dev@7a91017` | Separate materials project |

## F) Clone Utilities (Not Method Repos)

| Local path | Remote | Branch@HEAD | Purpose |
|---|---|---|---|
| `clone-anonymous-github` | `https://github.com/fedebotu/clone-anonymous-github.git` | `main@95fb985` | Utility for cloning GitHub repos anonymously |
| `clone_anonymous_github` | `https://github.com/kynehc/clone_anonymous_github.git` | `main@429551a` | Similar clone utility |

## G) Recommended "Active Research Set" for PF

Use this subset as the primary comparison/borrowing pool:

1. `pf_surrogate_modelling` (main implementation)
2. `PBFM` (physics FM objective ideas)
3. `UniDB` + `RDBM` + `DiffusionBridge` (bridge objectives/schedules/samplers)
4. `dsbm-pytorch` (iterative bridge fitting contract)
5. `SPL_OFM` + `pcfm_upstream` (stochastic/operator and physics-constrained FM ideas)
6. `lola_upstream` (latent physics diffusion design choices)

## H) Practical Notes

1. `diffusers` and `diffusers_clean` are currently identical (`main@b712042da`). Keep one as canonical and treat the other as backup.
2. `DBFM-3E8E` and `diffusion_bridge_unified_framework` are important but not git-tracked in-place; avoid assuming upstream parity.
3. For any benchmark write-up, cite both local path and upstream remote+commit for reproducibility.
