#!/usr/bin/env python3
# /scratch/project_2008261/solidification_modelling/GG_project/tdxdsurrogate2/scripts/smoketest_train.py
# Overfit on 3 consecutive pairs from one simulation. No second-person pronouns in code or comments.

# Ensure repository root is on sys.path so local packages are importable.
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, re, json, glob
from time import perf_counter
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt

# PhysicsNeMo: optimisation wrapper and metric (kept). Distributed and launch utils removed.
from physicsnemo.utils import StaticCaptureTraining
from physicsnemo.metrics.general.mse import mse as pn_mse

try:
    from models.datapipes.h5_dataset_solid import RSPairsDataset
except ModuleNotFoundError:
    RSPairsDataset = None
from models.backbones.uafno_cond import UAFNO_PreSkip_Full


def _collate(batch):
    x = torch.stack([b["x"] for b in batch], 0)
    y = torch.stack([b["y"] for b in batch], 0)
    c = torch.stack([b["cond"] for b in batch], 0)
    return {"x": x, "y": y, "cond": c}


def _resolve_group_by_original_name(h5_path: str, sim_hint: str) -> str:
    base = os.path.dirname(h5_path)
    meta_path = os.path.join(base, "sim_meta.json")
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
        # meta: {gid: {"original_name": "...", ...}}
        for gid, info in meta.items():
            name = str(info.get("original_name", ""))
            if sim_hint in name:
                return gid
    # Fallback: use the only group in the file
    import h5py
    with h5py.File(h5_path, "r") as h5:
        keys = sorted(list(h5.keys()))
    if len(keys) == 1:
        return keys[0]
    raise RuntimeError(f"Could not resolve group for hint={sim_hint!r}.")


def _restrict_to_pairs(ds: RSPairsDataset, gid: str, pairs_steps: List[Tuple[int, int]]) -> None:
    """Filter ds.index to rows from gid where pairs_time equals any requested (ti,tj)."""
    import h5py
    h5: h5py.File = ds.h5
    if gid not in h5:
        raise KeyError(f"Group {gid} not in HDF5.")
    g = h5[gid]
    if "pairs_time" not in g:
        raise RuntimeError("pairs_time dataset missing; cannot filter by absolute steps.")
    pt = g["pairs_time"][...].astype(np.int64)  # (P,2)
    want = set((int(a), int(b)) for a, b in pairs_steps)
    keep_rows = [i for i, (a, b) in enumerate(pt.tolist()) if (a, b) in want]
    if not keep_rows:
        raise RuntimeError("Requested pairs not found in pairs_time.")

    # Rebuild global index to only these rows
    ds.index = [(gid, int(i)) for i in keep_rows]

    # Recompute Δt cache and stats on the kept rows
    eff_dt = float(g.attrs.get("effective_dt", np.nan))
    dstep = (pt[keep_rows, 1] - pt[keep_rows, 0]).astype(np.int64)
    if ds.use_seconds and np.isfinite(eff_dt) and eff_dt > 0.0:
        dt_vec = dstep.astype(np.float64) * eff_dt
    else:
        dt_vec = dstep.astype(np.float64)

    ds._dt = dt_vec.copy()  # cache aligned with ds.index
    ds.dt_mu = float(dt_vec.mean())
    sd = float(dt_vec.std())
    ds.dt_sd = sd if sd > 0.0 else 1.0
    ds.dt_min = float(dt_vec.min())
    ds.dt_max = float(dt_vec.max())


def _ckpt_latest(path_dir: pathlib.Path) -> Optional[pathlib.Path]:
    files = sorted(path_dir.glob("epoch_*.pt"))
    if not files:
        return None
    # Sort by epoch number
    def _ep(p: pathlib.Path) -> int:
        m = re.search(r"epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    files.sort(key=_ep)
    return files[-1]


def _save_training_state(path_dir: pathlib.Path, epoch: int, model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                         model_kwargs: Dict) -> pathlib.Path:
    path_dir.mkdir(parents=True, exist_ok=True)
    f = path_dir / f"epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_kwargs": model_kwargs,
        },
        f,
    )
    return f


def _load_training_state(path_dir: pathlib.Path, model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                         device: torch.device) -> int:
    ck = _ckpt_latest(path_dir)
    if ck is None:
        return 0
    data = torch.load(ck, map_location=device)
    model.load_state_dict(data["state_dict"])
    try:
        optimizer.load_state_dict(data["optimizer"])
        scheduler.load_state_dict(data["scheduler"])
    except Exception:
        pass
    return int(data.get("epoch", 0)) + 1


class RSModelTester:
    """Utility to load a saved checkpoint, evaluate on a dataset, and save visualisations."""
    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None):
        self.ckpt_path = ckpt_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.meta: Optional[Dict] = None

    def load(self) -> None:
        data = torch.load(self.ckpt_path, map_location=self.device)
        self.meta = {
            "model_class": data.get("model_class", "UAFNO_PreSkip_Full"),
            "model_kwargs": data["model_kwargs"],
        }
        self.model = UAFNO_PreSkip_Full(**self.meta["model_kwargs"]).to(self.device)
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, ds: RSPairsDataset, batch_size: int = 3) -> dict:
        assert self.model is not None, "Call load() before evaluate()."
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0,
            pin_memory=torch.cuda.is_available(), collate_fn=_collate, drop_last=False
        )
        mse_sum, mae_sum, n = 0.0, 0.0, 0
        for batch in loader:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            cond = batch["cond"].to(self.device, non_blocking=True)
            yhat = self.model(x, cond)
            mse_sum += pn_mse(yhat, y).item() * y.numel()
            mae_sum += torch.abs(yhat - y).sum().item()
            n += y.numel()
        return {"mse": mse_sum / n, "mae": mae_sum / n}

    @torch.no_grad()
    def visualise_batch(self, ds: RSPairsDataset, save_path: str, idx: int = 0) -> None:
        """Save a figure showing target, prediction, and absolute error per channel for one sample."""
        assert self.model is not None, "Call load() before visualise_batch()."
        sample = ds[idx]
        x = sample["x"].unsqueeze(0).to(self.device)
        y = sample["y"].unsqueeze(0).to(self.device)
        cond = sample["cond"].unsqueeze(0).to(self.device)
        yhat = self.model(x, cond)

        y_np = y[0].detach().cpu().numpy()
        yhat_np = yhat[0].detach().cpu().numpy()
        err_np = np.abs(yhat_np - y_np)

        C = y_np.shape[0]
        cols = 3
        fig, axes = plt.subplots(nrows=C, ncols=cols, figsize=(3.2 * cols, 2.8 * C), squeeze=False)
        for c in range(C):
            for j, img in enumerate([y_np[c], yhat_np[c], err_np[c]]):
                ax = axes[c, j]
                im = ax.imshow(img, origin="lower")
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_title(["target", "prediction", "abs error"][j])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if RSPairsDataset is None:
        raise ImportError("RSPairsDataset unavailable; add models.datapipes.h5_dataset_solid before using this smoke test.")

    # Paths
    train_h5 = "/scratch/project_2008261/pf_surrogate_modelling/data/deterministic/simulation_train.h5"
    script_dir = pathlib.Path(__file__).resolve().parent
    ckpt_model_best = script_dir / "smoketest_uafno_best.pt"       # Pure PyTorch artefact
    ckpt_training_dir = script_dir / "checkpoints"                  # Resumable training state
    vis_png = script_dir / "smoketest_eval.png"
    ckpt_training_dir.mkdir(parents=True, exist_ok=True)

    # Resolve target simulation group
    gid = _resolve_group_by_original_name(train_h5, sim_hint="1.4")

    # Dataset
    train_ds = RSPairsDataset(
        path=train_h5,
        groups=[gid],
        normalise_dt="zscore",
        use_seconds=True,
        use_zscore_G=False,
        enforce_monotonic_time=True,
        nbins_dt_hist=0,
        store_dt_vector=False,
    )

    # Keep only specific pairs at Euler steps
    requested = [(273000, 274000), (274000, 275000), (275000, 276000)]
    _restrict_to_pairs(train_ds, gid, requested)

    # Shapes for AFNO bottleneck (four downsamples → /16)
    sample = train_ds[0]
    C, H, W = sample["x"].shape
    afno_h, afno_w = H // 16, W // 16

    # Model
    model_kwargs = dict(
        n_channels=C,
        n_classes=C,
        in_factor=32,
        cond_dim=2,
        afno_inp_shape=(afno_h, afno_w),
        afno_depth=8,
        afno_mlp_ratio=8.0,
    )
    model = UAFNO_PreSkip_Full(**model_kwargs).to(device)
    model.train()

    # Loader over the 3 selected pairs
    loader = DataLoader(
        train_ds,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        drop_last=False,
    )

    # Optimiser and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=20,
        threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8
    )

    # Static-captured training step (does backward() + optimiser.step() internally)
    @StaticCaptureTraining(model=model, optim=opt, cuda_graph_warmup=11)
    def training_step(x, cond, y):
        yhat = model(x, cond)
        loss = pn_mse(yhat, y)
        return loss

    # Resume from latest local checkpoint if present
    start_ep = _load_training_state(ckpt_training_dir, model, opt, sched, device)

    # Overfit loop with local checkpointing
    epochs = 200
    best = float("inf")
    data_iter = iter(loader)
    for ep in range(start_ep, epochs):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)

        t0 = perf_counter()
        loss = training_step(x, cond, y)
        dt_ms = (perf_counter() - t0) * 1e3

        val_loss = float(loss.item())
        lr_before = [g["lr"] for g in opt.param_groups]
        sched.step(val_loss)
        lr_after = [g["lr"] for g in opt.param_groups]
        lr_now = lr_after[0]
        lr_reduced = any(a < b for a, b in zip(lr_after, lr_before))

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "model_class": "UAFNO_PreSkip_Full",
                    "best_val_loss": best,
                },
                ckpt_model_best,
            )

        if ep % 10 == 0 or ep == epochs - 1:
            _save_training_state(ckpt_training_dir, ep, model, opt, sched, model_kwargs)

        if ep % 10 == 0:
            tag = "↓lr" if lr_reduced else ""
            print(f"epoch={ep:04d} loss={val_loss:.6e} lr={lr_now:.3e} time_ms={dt_ms:.2f} {tag}")

    # Report Δt stats for the restricted set
    print(
        f"Kept pairs: {requested}  |  Δt stats μ={train_ds.dt_mu:.6g}, σ={train_ds.dt_sd:.6g}, "
        f"min={train_ds.dt_min:.6g}, max={train_ds.dt_max:.6g}"
    )

    # Evaluate and visualise using the best model artefact
    tester = RSModelTester(str(ckpt_model_best), device=device)
    tester.load()
    metrics = tester.evaluate(train_ds, batch_size=3)
    print(f"Evaluation on restricted set: MSE={metrics['mse']:.6e}  MAE={metrics['mae']:.6e}")
    tester.visualise_batch(train_ds, save_path=str(vis_png), idx=0)
    print(f"Saved: best_model={ckpt_model_best}  training_checkpoints_dir={ckpt_training_dir}  vis_png={vis_png}")

    train_ds.close()


if __name__ == "__main__":
    main()
