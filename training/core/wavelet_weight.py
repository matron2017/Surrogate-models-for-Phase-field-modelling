# losses/wavelet_importance.py
import torch
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward  # type: ignore[import]
except Exception:
    DWTForward = None  # type: ignore[assignment]

__all__ = [
    "wavelet_importance_per_channel",
    "wavelet_importance_weighted_mse",
]

_DWT_CACHE: dict[tuple, torch.nn.Module] = {}


def _get_dwt_forward(J, wave, mode, device, dtype):
    if DWTForward is None:
        raise ImportError(
            "pytorch_wavelets is required for wavelet_importance_per_channel "
            "but could not be imported."
        )
    key = (J, wave, mode, str(device), str(dtype))
    dwt = _DWT_CACHE.get(key)
    if dwt is None:
        dwt = DWTForward(J=J, wave=wave, mode=mode)
        # Optional: pass separable=False if GPU is faster for the naive impl :contentReference[oaicite:1]{index=1}
        dwt = dwt.to(device=device, dtype=dtype)
        _DWT_CACHE[key] = dwt
    else:
        dwt = dwt.to(device=device, dtype=dtype)
        _DWT_CACHE[key] = dwt
    return dwt


def wavelet_importance_per_channel(
    y: torch.Tensor,
    J: int = 1,
    wave: str = "haar",
    mode: str = "zero",
    theta: float = 0.8,
    alpha: float = 1.25,
    beta_w: float = 6.0,
    eps: float = 1e-8,
    flat_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if DWTForward is None:
        raise ImportError(
            "pytorch_wavelets is required for wavelet_importance_per_channel "
            "but could not be imported."
        )

    assert y.dim() == 4, "Expected y with shape [B,C,H,W]"
    B, C, H, W = y.shape
    device, dtype = y.device, y.dtype

    dwt = _get_dwt_forward(J, wave, mode, device, dtype)

    Yl, Yh = dwt(y)  # Yh: list of [B,C,3,Hj,Wj]

    F_accum = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    for h in Yh:
        F_j = (h * h).sum(dim=2)  # [B,C,Hj,Wj]
        F_j_up = F.interpolate(F_j, size=(H, W), mode="bilinear", align_corners=False)
        F_accum = F_accum + F_j_up

    F_flat = F_accum.view(B, C, -1)
    F_min = F_flat.min(dim=-1, keepdim=True).values
    F_max = F_flat.max(dim=-1, keepdim=True).values

    flat_span = (F_max - F_min)
    flat_mask = flat_span <= flat_eps          # [B,C,1]

    q = torch.quantile(F_flat, theta, dim=-1, keepdim=True)  # [B,C,1]

    q_b = q.view(B, C, 1, 1)
    Fmax_b = F_max.view(B, C, 1, 1)

    denom = (Fmax_b - q_b).clamp_min(eps)
    num = (F_accum - q_b).clamp_min(0.0)
    high = F_accum > q_b

    a = torch.ones_like(F_accum)
    a = torch.where(
        high,
        alpha + (beta_w - alpha) * (num / denom),
        a,
    )

    if flat_mask.any():
        flat_bc = flat_mask.view(B, C, 1, 1)
        a = torch.where(flat_bc, torch.ones_like(a), a)

    return a, F_accum


def wavelet_importance_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_ref: torch.Tensor | None = None,
    J: int = 1,
    wave: str = "haar",
    mode: str = "zero",
    theta: float = 0.8,
    alpha: float = 1.25,
    beta_w: float = 6.0,
) -> torch.Tensor:
    assert pred.shape == target.shape, "pred and target must have the same shape"
    if y_ref is None:
        y_ref = target
    assert y_ref.shape == pred.shape, "y_ref must have the same shape as pred/target"

    with torch.no_grad():
        a_w, _ = wavelet_importance_per_channel(
            y_ref,
            J=J,
            wave=wave,
            mode=mode,
            theta=theta,
            alpha=alpha,
            beta_w=beta_w,
        )

    r = pred - target
    return (a_w * (r * r)).mean()
