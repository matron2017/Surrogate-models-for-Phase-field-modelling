# Tools Index

- `watch_slurm_job.py` – Polls `squeue`/`sacct` to report job status until completion (add `--interval` to change cadence).
- `chain_wavelet_and_train.py` – Submits a wavelet precompute job, waits, then submits training (see `--wavelet-sbatch`, `--train-sbatch`).
- `codex_dir_check.sh` – Prints a trimmed tree and doc headers to refresh repo context.
- `print_tree.py` – Lightweight tree printer used by Codex helpers.
