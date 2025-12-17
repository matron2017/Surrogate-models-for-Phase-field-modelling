from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PhaseFieldDataset
from models_vae import build_phasefield_vae
from wavelet_loss import wavelet_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AutoencoderKL on PF frames")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/diffusion_prototype/configs/vae_config.json"),
        help="Path to the VAE config JSON",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    dataset = PhaseFieldDataset(
        h5_path=data_cfg["h5_path"],
        dataset_key=data_cfg.get("dataset_key", "images"),
        scalar_key=data_cfg.get("scalar_key"),
        pair_mode=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    vae = build_phasefield_vae(cfg.get("vae")).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=float(train_cfg.get("lr", 3e-4)))

    out_dir = Path(train_cfg.get("out_dir", "experiments/diffusion_prototype/runs/vae"))
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_wave = float(loss_cfg.get("lambda_wavelet", 0.1))
    lambda_kl = float(loss_cfg.get("lambda_kl", 1e-4))

    global_step = 0
    for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
        vae.train()
        for batch in loader:
            global_step += 1
            frame = batch["frame"].to(device)
            posterior = vae.encode(frame).latent_dist
            latents = posterior.sample() * vae.config.scaling_factor
            recon = vae.decode(latents).sample

            mse = F.mse_loss(recon, frame)
            wav = wavelet_loss(recon, frame)
            kl = posterior.kl().mean()
            loss = mse + lambda_wave * wav + lambda_kl * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % int(train_cfg.get("log_every", 10)) == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} mse={mse.item():.4f} wav={wav.item():.4f} kl={kl.item():.4f}",
                    flush=True,
                )

        ckpt_path = out_dir / f"vae_epoch{epoch:03d}.pt"
        torch.save(vae.state_dict(), ckpt_path)
        print(f"[checkpoint] Saved {ckpt_path}")


if __name__ == "__main__":
    main()
