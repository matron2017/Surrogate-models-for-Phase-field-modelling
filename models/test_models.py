# tests/test_models_debug.py
# Debug and learning checks + comprehensive model summary.
# No second-person phrasing in comments.

import sys, math, json, csv, io, time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# optional: add project root if needed (parent of 'models')
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.backbones.unet_conv_att_cond import UNet_SSA_PreSkip_Full
from models.backbones.uafno_cond import UAFNO_PreSkip_Full
#/scratch/project_2008261/solidification_modelling/models/rapid_solidification/uafno_cond.py
#/scratch/project_2008261/solidification_modelling/GG_project/tdxdsurrogate2/models/rapid_solidification/unet_conv_att_cond.py

def tensor_stats(t: torch.Tensor) -> str:
    t = t.detach()
    return f"shape={tuple(t.shape)}, min={t.min().item():.4g}, max={t.max().item():.4g}, mean={t.mean().item():.4g}, std={t.std(unbiased=False).item():.4g}"


class ShapeTracer:
    """Collects and prints output tensor shapes per selected modules via forward hooks."""
    def __init__(self, root: nn.Module):
        self.root = root
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, m: nn.Module, inp, out):
        def _shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape), x.dtype, str(x.device)
            return (), torch.float32, "n/a"
        if isinstance(out, (tuple, list)):
            shapes = [_shape(t)[0] for t in out if isinstance(t, torch.Tensor)]
            dtype = next((t.dtype for t in out if isinstance(t, torch.Tensor)), torch.float32)
            device = next((str(t.device) for t in out if isinstance(t, torch.Tensor)), "cpu")
            shape = ("multi",) + tuple(shapes)
        else:
            shape, dtype, device = _shape(out)
        name = None
        for n, mm in self.root.named_modules():
            if mm is m:
                name = n or self.root.__class__.__name__
                break
        name = name or m.__class__.__name__
        print(f"[shape] {name:<40s} -> {shape}  {dtype}  {device}")

    def __enter__(self):
        for m in self.root.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm,
                              nn.ReLU, nn.GELU, nn.Upsample, nn.MaxPool2d, nn.AvgPool2d, nn.Identity, nn.Linear)) \
               or any(k in m.__class__.__name__.lower() for k in ("attention", "afno", "fourier", "fno")):
                self._hooks.append(m.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._hooks:
            h.remove()
        return False


# ---- FLOPs / MACs / activation accounting ----

@dataclass
class ModuleReport:
    name: str
    type: str
    out_shape: Tuple[int, ...]
    params: int
    macs: float
    flops: float
    act_elems: int
    act_bytes: int

class SummaryCollector:
    """Estimates per-module MACs/FLOPs and activation memory via forward hooks."""
    def __init__(self, root: nn.Module):
        self.root = root
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.reports: List[ModuleReport] = []
        self.param_counts: Dict[str, int] = {n: int(p.numel()) for n, p in root.named_parameters(recurse=True)}
        self.param_ownermap: Dict[nn.Module, int] = {}
        for m in root.modules():
            cnt = 0
            for n, p in m.named_parameters(recurse=False):
                cnt += int(p.numel())
            self.param_ownermap[m] = cnt

    @staticmethod
    def _numel(x: torch.Tensor) -> int:
        return int(x.numel())

    @staticmethod
    def _dtype_bytes(dtype: torch.dtype) -> int:
        return {
            torch.float64: 8, torch.float32: 4, torch.float16: 2,
            torch.bfloat16: 2, torch.int64: 8, torch.int32: 4,
            torch.int16: 2, torch.int8: 1, torch.bool: 1
        }.get(dtype, 4)

    @staticmethod
    def _conv2d_macs(m: nn.Conv2d, out: torch.Tensor) -> float:
        # MACs per output element = (Cin/groups) * Kh * Kw
        cin_g = m.in_channels // m.groups
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        macs_per_out = cin_g * kh * kw
        return macs_per_out * out.numel()

    @staticmethod
    def _convtranspose2d_macs(m: nn.ConvTranspose2d, out: torch.Tensor) -> float:
        cin_g = m.in_channels // m.groups
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        macs_per_out = cin_g * kh * kw
        return macs_per_out * out.numel()

    @staticmethod
    def _linear_macs(m: nn.Linear, out: torch.Tensor, inp: torch.Tensor) -> float:
        # inp shape (..., in_features), out shape (..., out_features)
        batch = int(inp.numel() // m.in_features)
        return batch * m.in_features * m.out_features

    @staticmethod
    def _bn_ops(out: torch.Tensor) -> float:
        # Approximate: 2 ops per element (scale + shift)
        return 2.0 * out.numel()

    @staticmethod
    def _eltwise_ops(out: torch.Tensor) -> float:
        # ReLU/GELU etc. Approximate 1 op/element.
        return float(out.numel())

    @staticmethod
    def _pool_ops(m: nn.Module, out: torch.Tensor) -> float:
        # Max/Average pooling approximate: kernel comparisons/adds per output.
        k = getattr(m, "kernel_size", (1, 1))
        if isinstance(k, tuple): kh, kw = k
        else: kh = kw = k
        return float(max(0, kh * kw - 1) * out.numel())

    @staticmethod
    def _fft2_flops(c: int, h: int, w: int, transforms: int = 2) -> float:
        # Very rough: 5*N*log2(N) per 1D FFT; 2D ~ 5*H*W*(log2 H + log2 W).
        # Multiply by number of transforms (forward+inverse).
        return float(c * transforms * 5.0 * h * w * (math.log2(max(2, h)) + math.log2(max(2, w))))

    @staticmethod
    def _attention_macs(seq_len: int, head_dim: int, heads: int = 1) -> float:
        # QK^T and Attn*V. Count MACs. Softmax omitted.
        return float(2.0 * heads * seq_len * seq_len * head_dim)

    def _hook(self, m: nn.Module, inp, out):
        # Determine name
        name = None
        for n, mm in self.root.named_modules():
            if mm is m:
                name = n or self.root.__class__.__name__
                break
        name = name or m.__class__.__name__
        mtype = m.__class__.__name__

        # Normalise out tensor
        if isinstance(out, (tuple, list)):
            t = next((x for x in out if isinstance(x, torch.Tensor)), None)
            if t is None:
                return
            out_t = t
        else:
            out_t = out
        out_shape = tuple(out_t.shape)
        dtype_bytes = self._dtype_bytes(out_t.dtype)
        act_elems = self._numel(out_t)
        act_bytes = act_elems * dtype_bytes

        # Params owned by this module (non-recursive)
        params_here = self.param_ownermap.get(m, 0)

        macs = 0.0
        flops = 0.0

        try:
            if isinstance(m, nn.Conv2d):
                macs = self._conv2d_macs(m, out_t)
            elif isinstance(m, nn.ConvTranspose2d):
                macs = self._convtranspose2d_macs(m, out_t)
            elif isinstance(m, nn.Linear):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                macs = self._linear_macs(m, out_t, x)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                flops = self._bn_ops(out_t)
            elif isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.Identity)):
                flops = self._eltwise_ops(out_t)
            elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
                flops = self._pool_ops(m, out_t)

            # Heuristics for AFNO/Fourier blocks
            lc = m.__class__.__name__.lower()
            if any(k in lc for k in ("afno", "fourier", "fno")):
                _, c, h, w = out_t.shape
                flops += self._fft2_flops(c, h, w, transforms=2)

            # Heuristics for simple SSA blocks with q/k/v projections attached
            if hasattr(m, "q_proj") and hasattr(m, "k_proj") and hasattr(m, "v_proj"):
                # Add 1x1 conv costs already covered by Conv2d hooks; only add attention matmul.
                _, cq, h, w = getattr(m, "q_proj").weight.shape  # (out_c, in_c, 1,1)
                heads = int(getattr(m, "heads", 1))
                head_dim = max(1, cq // heads)
                L = int(h * w) if isinstance(out_t, torch.Tensor) else 0
                macs += self._attention_macs(L, head_dim, heads=heads)
        except Exception:
            pass

        if flops == 0.0:
            flops = 2.0 * macs  # default: 1 MAC = 2 FLOPs

        self.reports.append(ModuleReport(
            name=name, type=mtype, out_shape=out_shape,
            params=params_here, macs=float(macs), flops=float(flops),
            act_elems=act_elems, act_bytes=act_bytes,
        ))

    def __enter__(self):
        for m in self.root.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm,
                              nn.ReLU, nn.GELU, nn.Upsample, nn.MaxPool2d, nn.AvgPool2d, nn.Identity, nn.Linear)) \
               or any(k in m.__class__.__name__.lower() for k in ("attention", "afno", "fourier", "fno")) \
               or hasattr(m, "q_proj"):
                self._hooks.append(m.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._hooks:
            h.remove()
        return False

    # Aggregate helpers
    def totals(self) -> Dict[str, Any]:
        tot_params = sum(r.params for r in self.reports)
        tot_macs = sum(r.macs for r in self.reports)
        tot_flops = sum(r.flops for r in self.reports)
        peak_act_bytes = max((r.act_bytes for r in self.reports), default=0)
        return dict(params=tot_params, macs=tot_macs, flops=tot_flops, peak_act_bytes=peak_act_bytes)

    # Output styles
    def to_markdown(self) -> str:
        rows = ["| module | type | out_shape | params | MACs | FLOPs | act_mem |",
                "|:--|:--|:--|--:|--:|--:|--:|"]
        for r in self.reports:
            rows.append(f"| {r.name} | {r.type} | {list(r.out_shape)} | {r.params} | "
                        f"{r.macs:.3e} | {r.flops:.3e} | {r.act_bytes/1e6:.2f} MB |")
        t = self.totals()
        rows.append(f"| **TOTAL** |  |  | **{t['params']}** | **{t['macs']:.3e}** | **{t['flops']:.3e}** | **{t['peak_act_bytes']/1e6:.2f} MB** |")
        return "\n".join(rows)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["module", "type", "out_shape", "params", "MACs", "FLOPs", "act_bytes"])
        for r in self.reports:
            w.writerow([r.name, r.type, list(r.out_shape), r.params, f"{r.macs:.6e}", f"{r.flops:.6e}", r.act_bytes])
        t = self.totals()
        w.writerow(["TOTAL", "", "", t["params"], f"{t['macs']:.6e}", f"{t['flops']:.6e}", t["peak_act_bytes"]])
        return buf.getvalue()

    def to_json(self) -> str:
        return json.dumps({
            "layers": [dict(name=r.name, type=r.type, out_shape=list(r.out_shape),
                            params=r.params, macs=r.macs, flops=r.flops,
                            act_elems=r.act_elems, act_bytes=r.act_bytes)
                       for r in self.reports],
            "totals": self.totals()
        }, indent=2)


def snapshot_param_norms(model: nn.Module) -> Dict[str, Tuple[float, bool]]:
    out = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            out[n] = (float(p.norm().item()), bool(p.requires_grad))
    return out


def grad_norms(model: nn.Module) -> Dict[str, float]:
    out = {}
    for n, p in model.named_parameters():
        out[n] = 0.0 if p.grad is None else float(p.grad.norm().item())
    return out


@dataclass
class PerfTimes:
    fw_ms: float
    bw_ms: float
    step_ms: float
    throughput_mpix_s: float


def timed_step(model: nn.Module, loss: torch.Tensor, pixels: int) -> PerfTimes:
    dev = next(model.parameters()).device
    use_cuda = dev.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(True); e2 = torch.cuda.Event(True); e3 = torch.cuda.Event(True)
        e0.record()
    t0 = time.perf_counter()

    loss.backward()
    if use_cuda:
        e1.record()
    t1 = time.perf_counter()

    # optimiser step is done outside; here just measure backward by default
    if use_cuda:
        torch.cuda.synchronize()
        fw_ms = 0.0  # forward measured separately outside
        bw_ms = e0.elapsed_time(e1)
        step_ms = 0.0
    else:
        fw_ms = 0.0
        bw_ms = (t1 - t0) * 1e3
        step_ms = 0.0
    throughput = (pixels / 1e6) / max(bw_ms, 1e-6) * 1e3
    return PerfTimes(fw_ms=fw_ms, bw_ms=bw_ms, step_ms=step_ms, throughput_mpix_s=throughput)


def run_debug(model: nn.Module, H: int, W: int, Cin: int, Cout: int, cond_dim: int, device: torch.device, style: str = "all") -> None:
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(17)

    x = torch.randn(1, Cin, H, W, device=device)
    cond = torch.randn(1, cond_dim, device=device)
    target = torch.randn(1, Cout, H, W, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Forward + tracing
    model.train()
    with ShapeTracer(model), SummaryCollector(model) as summ:
        y = model(x, cond)
        # Aggregate + print after forward to ensure reports are populated
        totals = summ.totals()
        md = summ.to_markdown()
        csv_text = summ.to_csv()
        json_text = summ.to_json()

    print("final output:", tensor_stats(y))
    loss = F.mse_loss(y, target)
    print(f"loss: {loss.item():.6f}")

    # Parameter stats
    before = snapshot_param_norms(model)
    optim.zero_grad(set_to_none=True)

    # Timed backward
    perf = timed_step(model, loss, pixels=H * W)
    gnorms = grad_norms(model)
    zero_grad_params = [n for n, g in gnorms.items() if g == 0.0 and before[n][1]]
    if len(zero_grad_params) > 0:
        print(f"params with zero grad ({len(zero_grad_params)}):")
        for n in zero_grad_params[:20]:
            print("  ", n)
        if len(zero_grad_params) > 20:
            print("  ...")

    optim.step()
    after = snapshot_param_norms(model)

    print("\nparameter update summary (first 30):")
    for i, (n, (norm0, req)) in enumerate(before.items()):
        if i >= 30: break
        norm1 = after[n][0]
        delta = abs(norm1 - norm0)
        print(f"{n:60s} | grad={gnorms[n]:.4e} | Δ||w||={delta:.4e} | req={req}")

    # Model-level summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n=== Model summary ===")
    print(f"params_total={total_params:,}  params_trainable={trainable_params:,}")
    print(f"MACs≈{totals['macs']:.3e}  FLOPs≈{totals['flops']:.3e}  peak_act_mem≈{totals['peak_act_bytes']/1e6:.2f} MB")
    print(f"backward_time_ms≈{perf.bw_ms:.2f}  throughput≈{perf.throughput_mpix_s:.2f} MPix/s")

    # Output styles
    if style in ("all", "markdown"):
        print("\n--- layerwise (markdown) ---\n" + md)
    if style in ("all", "csv"):
        print("\n--- layerwise (csv) ---\n" + csv_text)
    if style in ("all", "json"):
        print("\n--- layerwise (json) ---\n" + json_text)


def main():
    H, W = 1024, 1024
    Cin, Cout, cond_dim = 2, 2, 2
    assert H % 16 == 0 and W % 16 == 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # UNet + SSA
    m_unet = UNet_SSA_PreSkip_Full(
        n_channels=Cin, n_classes=Cout, in_factor=78, cond_dim=cond_dim,
        afno_inp_shape=(H // 16, W // 16), ssa_heads=24, ssa_qk_ratio=1/2
    ).to(device)
    print("\n=== UNet_SSA_PreSkip_Full ===")
    run_debug(m_unet, H, W, Cin, Cout, cond_dim, device, style="all")

    # U-Net + AFNO
    m_uafno = UAFNO_PreSkip_Full(
        n_channels=Cin, n_classes=Cout, in_factor=52, cond_dim=cond_dim,
        afno_inp_shape=(H // 16, W // 16), afno_depth=4, afno_mlp_ratio=8.0
    ).to(device)
    print("\n=== UAFNO_PreSkip_Full ===")
    run_debug(m_uafno, H, W, Cin, Cout, cond_dim, device, style="all")


if __name__ == "__main__":
    main()
