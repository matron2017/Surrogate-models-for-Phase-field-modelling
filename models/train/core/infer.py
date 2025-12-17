#!/usr/bin/env python3
import argparse, json, yaml, torch
from pathlib import Path
from torch.amp import autocast

def _load_symbol(py_path: str, symbol: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(Path(py_path).stem, py_path)
    if spec is None or spec.loader is None: raise ImportError(f"Failed to load: {py_path}")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, symbol): raise AttributeError(f"Symbol '{symbol}' not found in {py_path}")
    return getattr(mod, symbol)

def load_bundle(bundle_dir: Path, map_location: str = "cpu", strict: bool = True):
    model_cfg = yaml.safe_load((bundle_dir/"model_init.yaml").read_text())
    preproc   = yaml.safe_load((bundle_dir/"preproc.yaml").read_text())
    manifest  = json.loads((bundle_dir/"manifest.json").read_text())

    ModelClass = _load_symbol(model_cfg["file"], model_cfg["class"])
    model = ModelClass(**model_cfg.get("params", {}))
    ckpt = torch.load(bundle_dir/"checkpoint.best.pth", map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=strict)
    model.to(map_location).eval()
    return model, preproc, manifest

def run_once(model, x, cond=None, device="cpu", amp_dtype="bf16"):
    model.to(device)
    x = x.to(device)
    cond = cond.to(device) if cond is not None else None
    dtype = torch.bfloat16 if str(amp_dtype).lower()=="bf16" else torch.float16
    with torch.inference_mode(), autocast(device_type="cuda", enabled=(device=="cuda"), dtype=dtype):
        y = model(x, cond) if cond is not None else model(x)
    return y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    args = ap.parse_args()

    model, preproc, manifest = load_bundle(args.bundle, map_location=args.device)
    B = 1
    C_in = len(preproc["input_channels"]) if isinstance(preproc.get("input_channels"), (list, tuple)) else int(preproc.get("input_channels", 1))
    H, W = preproc["spatial_shape"]
    cd = int(preproc["conditioning"].get("cond_dim", 0)) if preproc["conditioning"].get("source","field").lower()=="channels" else 0
    x = torch.zeros(B, C_in + cd, H, W)
    cond = None if cd>0 else torch.zeros(B, int(preproc["conditioning"].get("cond_dim", 0)))
    y = run_once(model, x, cond=cond, device=args.device, amp_dtype=manifest.get("amp_dtype","bf16"))
    print("output", tuple(y.shape))
