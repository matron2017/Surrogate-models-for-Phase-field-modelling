from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from dataset import PhaseFieldDataset
from models_unet import build_phasefield_unet
from models_vae import build_phasefield_vae
from wavelet_loss import wavelet_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the latent diffusion surrogate")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/diffusion_prototype/configs/unet_config.json"),
        help="Config JSON driving the diffusion trainer",
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
        pair_mode=True,
        pair_strides=data_cfg.get("pair_stride", 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    vae = build_phasefield_vae(cfg.get("vae")).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae_ckpt = train_cfg.get("vae_checkpoint")
    if vae_ckpt:
        state = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(state)
        print(f"[vae] Loaded weights from {vae_ckpt}")

    unet = build_phasefield_unet(cfg.get("unet")).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=float(train_cfg.get("lr", 2e-4)))
    scheduler = DDPMScheduler(**cfg.get("scheduler", {}))

    out_dir = Path(train_cfg.get("out_dir", "experiments/diffusion_prototype/runs/diffusion"))
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_wave = float(loss_cfg.get("lambda_wavelet", 0.0))

    global_step = 0
    for epoch in range(1, int(train_cfg.get("epochs", 1)) + 1):
        unet.train()
        for batch in loader:
            global_step += 1
            current = batch["current"].to(device)
            target = batch["next"].to(device)

            with torch.no_grad():
                latent_curr = vae.encode(current).latent_dist.sample() * vae.config.scaling_factor
                latent_target = vae.encode(target).latent_dist.sample() * vae.config.scaling_factor

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (current.shape[0],), device=device).long()
            noise = torch.randn_like(latent_target)
            noisy_latents = scheduler.add_noise(latent_target, noise, timesteps)

            if "scalars" in batch:
                scalars = batch["scalars"].to(device)
                scalar_maps = scalars.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, scalars.shape[1], latent_curr.shape[-2], latent_curr.shape[-1]
                )
            else:
                scalar_maps = None

            unet_inputs = [noisy_latents, latent_curr]
            if scalar_maps is not None:
                unet_inputs.append(scalar_maps)
            unet_in = torch.cat(unet_inputs, dim=1)
            noise_pred = unet(unet_in, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            if lambda_wave > 0:
                with torch.no_grad():
                    pred_clean = scheduler.step(noise_pred, timesteps, noisy_latents).pred_original_sample
                    recon = vae.decode(pred_clean / vae.config.scaling_factor).sample
                loss += lambda_wave * wavelet_loss(recon, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % int(train_cfg.get("log_every", 25)) == 0:
                print(f"epoch={epoch} step={global_step} loss={loss.item():.4f}", flush=True)

        ckpt = out_dir / f"diffusion_epoch{epoch:03d}.pt"
        torch.save(unet.state_dict(), ckpt)
        print(f"[checkpoint] Saved {ckpt}")


if __name__ == "__main__":
    main()
