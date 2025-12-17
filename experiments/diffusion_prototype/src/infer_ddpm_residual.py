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
    parser = argparse.ArgumentParser(description="DDPM inference for residual AFNO patches")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/diffusion_prototype/configs/ddpm_placeholder.json"),
        help="Config JSON used for dataset/model instantiation",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=50)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def ddpm_sample(model, scheduler, xt, cond_vec, steps: int) -> torch.Tensor:
    device = xt.device
    residual = torch.randn_like(xt)
    scheduler.set_timesteps(steps, device=device)
    for t in scheduler.timesteps:
        residual_in = scheduler.scale_model_input(residual, t)
        model_in = torch.cat([residual_in, xt], dim=1)
        noise_pred = model(sample=model_in, timestep=t, cond_vec=cond_vec).sample
        residual = scheduler.step(noise_pred, t, residual).prev_sample
    return residual


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    scheduler_cfg = cfg.get("scheduler", {})
    train_cfg = cfg["training"]

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    dataset = ResidualPatchDataset(
        h5_path=data_cfg["h5_path"],
        dataset_key=data_cfg.get("dataset_key", "images"),
        scalar_key=data_cfg.get("scalar_key", "scalars"),
        patch_size=int(data_cfg.get("patch_size", 64)),
        pair_stride=int(data_cfg.get("pair_stride", 1)),
        seed=int(train_cfg.get("seed", 0)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    patch_size = int(data_cfg.get("patch_size", 64))
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

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    scheduler = DDPMScheduler(**scheduler_cfg)

    total_mse = 0.0
    n = 0
    for sample in loader:
        xt = sample["xt"].unsqueeze(0).to(device)
        residual_true = sample["residual"].unsqueeze(0).to(device)
        cond_vec = sample["cond"].unsqueeze(0).to(device)

        residual_pred = ddpm_sample(model, scheduler, xt, cond_vec, args.inference_steps)
        xt_pred = xt + residual_pred
        xt_true = xt + residual_true
        mse = F.mse_loss(xt_pred, xt_true).item()
        n += 1
        total_mse += mse
        print(f"sample={n} mse={mse:.6f}")
        if n >= args.num_samples:
            break

    if n:
        print(f"avg_mse={total_mse / n:.6f}")


if __name__ == "__main__":
    main()
