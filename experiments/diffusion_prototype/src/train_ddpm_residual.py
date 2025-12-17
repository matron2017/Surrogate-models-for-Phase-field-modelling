from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from dataset import ResidualPatchDataset
from models.backbones.uafno_diffusion import UAFNO_DiffusionUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AFNO diffusion residual model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/diffusion_prototype/configs/ddpm_placeholder.json"),
        help="Path to training config JSON",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def collate(batch):
    xt = torch.stack([item["xt"] for item in batch], dim=0)
    residual = torch.stack([item["residual"] for item in batch], dim=0)
    cond = torch.stack([item["cond"] for item in batch], dim=0)
    return {"xt": xt, "residual": residual, "cond": cond}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    scheduler_cfg = cfg.get("scheduler", {})

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ResidualPatchDataset(
        h5_path=data_cfg["h5_path"],
        dataset_key=data_cfg.get("dataset_key", "images"),
        scalar_key=data_cfg.get("scalar_key", "scalars"),
        patch_size=int(data_cfg.get("patch_size", 64)),
        pair_stride=int(data_cfg.get("pair_stride", 1)),
        seed=int(train_cfg.get("seed", 0)),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=True,
        collate_fn=collate,
    )

    patch_size = int(data_cfg.get("patch_size", 64))
    if patch_size % 16 != 0:
        raise ValueError("patch_size must be divisible by 16 for the AFNO bottleneck")
    afno_shape = tuple(model_cfg.get("afno_inp_shape", (patch_size // 16, patch_size // 16)))

    model = UAFNO_DiffusionUNet(
        n_channels=model_cfg.get("n_channels", 4),
        n_classes=model_cfg.get("n_classes", 2),
        in_factor=model_cfg.get("in_factor", 40),
        cond_dim=model_cfg.get("cond_dim", 2),
        afno_inp_shape=afno_shape,
        afno_depth=model_cfg.get("afno_depth", 12),
        num_blocks=model_cfg.get("num_blocks", 16),
        afno_mlp_ratio=model_cfg.get("afno_mlp_ratio", 12.0),
        time_embed_dim=model_cfg.get("time_embed_dim", 256),
        film_hidden=model_cfg.get("film_hidden", 128),
    ).to(device)

    noise_scheduler = DDPMScheduler(**scheduler_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    out_dir = Path(train_cfg.get("out_dir", "runs_debug/diffusion_smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(train_cfg.get("epochs", 1))
    log_every = int(train_cfg.get("log_every", 10))

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            global_step += 1
            xt = batch["xt"].to(device)
            residual = batch["residual"].to(device)
            cond_vec = batch["cond"].to(device)

            noise = torch.randn_like(residual)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (residual.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_residual = noise_scheduler.add_noise(residual, noise, timesteps)
            model_input = torch.cat([noisy_residual, xt], dim=1)
            noise_pred = model(sample=model_input, timestep=timesteps, cond_vec=cond_vec).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}", flush=True)

        ckpt_path = out_dir / f"epoch{epoch:03d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[checkpoint] saved {ckpt_path}")


if __name__ == "__main__":
    main()
