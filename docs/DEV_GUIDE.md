# Rapid Solidification — Dev Notes (Codex)

Quick checklist to keep runs reproducible, cheap, and comparable.

## Basics
- Set `PYTHONPATH=models` when running tests or scripts locally.
- Use the bundled env: `/scratch/project_2008261/physics_ml/bin/python3.11`.
- Absolute-time conditioning is already baked into the dataloader; no Δt fixes needed.

## Tests
- Fast backbone smoke: `PYTHONPATH=models python -m pytest tests/test_backbones_rs.py -q`.
- Full CPU suite: `PYTHONPATH=models python -m pytest tests -q` (no GPU needed).
- Add a new model? Copy an existing forward test in `tests/test_backbones_rs.py` with small shapes to avoid operator size errors.

## Training (gputest first)
- Submit to `gputest` for 10–20 min sanity runs, then scale to `gpu` for long jobs.
- UNet: `sbatch --partition=gputest --gres=gpu:v100:1 slurm/train_big_unet_att.sh`
- U-AFNO: `sbatch --partition=gputest --gres=gpu:v100:1 slurm/train_big_uafno.sh`
- FNO: `sbatch slurm/train_fno.sh` (already gputest-ready)
- Wavelet variants: set `CFG=/.../<model>_wavelet.yaml` after generating weights.

## Wavelet weights
- Generate once per setting: `python -m models.datapipes.precompute_wavelet_weights --h5 data/rapid_solidification/simulation_train.h5 --out data/rapid_solidification/simulation_train.wavelet.h5 --target-channels 0 1 --device cuda`.
- Heavy-weight variant (α=10000, β=50000, θ=0.75): `sbatch datapipes/wavelet_weights_a10000b50000.sh` (writes `simulation_train_a10000b50000.wavelet.h5`).
- Chain wavelet → train automatically: `python tools/chain_wavelet_and_train.py --wavelet-sbatch datapipes/wavelet_weights_a10000b50000.sh --train-sbatch slurm/train_big_uafno.sh --train-config configs/train/uafno_wavelet.yaml`. This waits for the wavelet job to finish, then submits training.

## Ablation parity (≈200 M params)
- UNet: `in_factor≈120`.
- U-AFNO: `in_factor≈48`, `afno_depth≈12`.
- FNO: `embed_channels≈208`, `depth≈8`, `modes≈12`.
- Keep conditioning `[t_abs_norm, thermal_norm]` across all configs.

## Experiment hygiene
- Track runs/metrics via MLflow or CSVs; keep configs in `configs/train/` under version control.
- Use modular plotting scripts under `visuals/` to compare checkpoints; prefer saving JSON/CSV metrics for diffing.
- Avoid checking in artefacts (`runs/`, `results/`, `.h5` weights); `.gitignore` already excludes them.

## Multi-GPU quick guide (≥2 GPUs, Puhti/Mahti)
- Request GPUs/CPUs that match torchrun: `--gres=gpu:v100:2` pairs with `--nproc_per_node=2`; keep `--cpus-per-task≈10×num_gpus` (e.g., 20 for 2 GPUs, 40 for 4 GPUs). Examples: `slurm/train_flowmatch_uafno_full_2gpu.sh` (2×V100) or bump to 4 GPUs by setting `--gres=gpu:v100:4`, `--cpus-per-task=40`, `--nproc_per_node=4`.
- Pin devices per rank inside the job: ensure `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))` is called (the trainer already wires LOCAL_RANK). Symptom of a bug: GPU0 at 100% while others idle.
- Avoid dataloader oversubscription: `world_size * num_workers_per_rank` should stay ≤ `cpus-per-task` (e.g., 2 GPUs × 6 workers each = 12 ≤ 20 CPUs is fine).
- Binding/visibility sanity check (first lines of the log): `srun -n1 --cpu-bind=cores --hint=nomultithread --gpu-bind=closest bash -lc 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi -L'`.
- Quick DDP check on a single node: `python -m models.train.core.ddp_multi_node_check --backend nccl --world-size 2 --nproc-per-node 2` inside the allocation; for Slurm, submit `slurm/test_parallel_multinode.sh` to exercise NCCL/env wiring before heavy runs.
