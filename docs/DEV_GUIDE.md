# Rapid Solidification — Dev Notes

Quick checklist to keep runs reproducible, cheap, and comparable.

## Basics
- Set `PYTHONPATH=models` when running tests or scripts locally.
- Use the bundled env: `/scratch/project_2008261/physics_ml/bin/python3.11`.
- Absolute-time conditioning is already baked into the dataloader; no Δt fixes needed.
- For the grouped thermal-latent pipeline view, use `docs/WORKFLOW_THERMAL_LATENT.md`.

## Tests
- Fast backbone smoke: `PYTHONPATH=models python -m pytest tests/test_backbones_rs.py -q`.
- Full CPU suite: `PYTHONPATH=models python -m pytest tests -q` (no GPU needed).
- Add a new model? Copy an existing forward test in `tests/test_backbones_rs.py` with small shapes to avoid operator size errors.

## Training (gputest first)
- Submit to `gputest` for 10–20 min sanity runs, then scale to `gpu` for long jobs.
- Active launchers are the AE-latent scripts in `slurm/` (AdamW/PSGD variants).
- Wavelet variants are optional/legacy; skip them unless you explicitly need wavelet-weighted losses.

## Wavelet weights (optional)
- Generate only when needed: `python -m models.datapipes.precompute_wavelet_weights --h5 <train_h5> --out <wavelet_out_h5> --target-channels 0 1 --device cuda`.
- Heavy-weight variant (alpha=10000, beta=50000, theta=0.75): `sbatch datapipes/wavelet_weights_a10000b50000.sh`.
- Optional chaining helper can be used with one of the active AE launchers.

## Ablation parity (≈200 M params)
- Historical UNet/UAFNO/FNO parity notes are retained for reference only.
- Current experiments use thermal-field-only conditioning: disable scalar conditioning and use `conditioning.use_theta` + `add_thermal`.

## Experiment hygiene
- Track runs/metrics via MLflow or CSVs; keep configs in `configs/train/` under version control.
- Use modular plotting scripts under `visuals/` to compare checkpoints; prefer saving JSON/CSV metrics for diffing.
- Avoid checking in artefacts (`runs/`, `results/`, `.h5` weights); `.gitignore` already excludes them.

## Multi-GPU quick guide (≥2 GPUs, Puhti/Mahti)
- Request GPUs/CPUs that match torchrun: `--gres=gpu:v100:2` pairs with `--nproc_per_node=2`; keep `--cpus-per-task≈10×num_gpus` (e.g., 20 for 2 GPUs, 40 for 4 GPUs).
- Pin devices per rank inside the job: ensure `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))` is called (the trainer already wires LOCAL_RANK). Symptom of a bug: GPU0 at 100% while others idle.
- Avoid dataloader oversubscription: `world_size * num_workers_per_rank` should stay ≤ `cpus-per-task` (e.g., 2 GPUs × 6 workers each = 12 ≤ 20 CPUs is fine).
- Binding/visibility sanity check (first lines of the log): `srun -n1 --cpu-bind=cores --hint=nomultithread --gpu-bind=closest bash -lc 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi -L'`.
- Quick DDP check on a single node: `python -m models.train.core.ddp_multi_node_check --backend nccl --world-size 2 --nproc-per-node 2` inside the allocation.
