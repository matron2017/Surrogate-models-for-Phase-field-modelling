#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DC_GEN_REPO_ROOT = Path(os.environ.get("DC_GEN_REPO_ROOT", PROJECT_ROOT / "external_refs" / "DC-Gen")).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DC_GEN_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DC_GEN_REPO_ROOT))

from scripts.train_dcae_finetune import PDEFieldDataset, pde_reconstruction_loss
from dc_gen.ae_model_zoo import DCAE_HF


def main() -> None:
    data_root = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
    out_dir = Path(os.environ.get("OUT_DIR", PROJECT_ROOT / "runs" / "autoencoder" / "forward_gputest"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] project_root={PROJECT_ROOT}")
    print(f"[env] dc_gen_repo={DC_GEN_REPO_ROOT}")
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)} mem_total={torch.cuda.get_device_properties(0).total_memory}")

    ds = PDEFieldDataset(str(data_root / "train.h5"), t_start=0, t_step=16, max_frames=2, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    x = next(iter(loader)).to(device)
    print(f"[data] dataset_len={len(ds)} batch_shape={tuple(x.shape)} dtype={x.dtype} min={float(x.min()):.6g} max={float(x.max()):.6g}")

    model_key = os.environ.get("DCAE_MODEL_KEY", "dc-ae-f32c32-in-1.0")
    model_source = os.environ.get("MODEL_SOURCE", f"mit-han-lab/{model_key}")
    print(f"[model] loading {model_source}")
    model = DCAE_HF.from_pretrained(model_source).to(device).eval()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[model] params={nparams}")

    with torch.no_grad():
        latent = model.encode(x)
        recon = model.decode(latent)
        losses = pde_reconstruction_loss(recon.float(), x.float(), lambda_grad=0.5, lambda_spec=0.1)

    print(f"[shape] input={tuple(x.shape)} latent={tuple(latent.shape)} recon={tuple(recon.shape)}")
    print("[loss] " + " ".join(f"{k}={float(v):.8g}" for k, v in losses.items()))
    payload = {
        "input_shape": list(x.shape),
        "latent_shape": list(latent.shape),
        "recon_shape": list(recon.shape),
        "input_dtype": str(x.dtype),
        "latent_dtype": str(latent.dtype),
        "recon_dtype": str(recon.dtype),
        "losses": {k: float(v) for k, v in losses.items()},
        "model_source": model_source,
        "params": int(nparams),
        "device": str(device),
    }
    (out_dir / "forward_shapes.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    assert tuple(x.shape) == (1, 3, 512, 512), tuple(x.shape)
    assert tuple(latent.shape) == (1, 32, 16, 16), tuple(latent.shape)
    assert tuple(recon.shape) == tuple(x.shape), (tuple(recon.shape), tuple(x.shape))
    print("[ok] forward shape smoke passed")


if __name__ == "__main__":
    main()
