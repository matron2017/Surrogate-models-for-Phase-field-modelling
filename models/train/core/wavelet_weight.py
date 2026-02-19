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
    "wavelet_bandpass_importance_per_channel",
    "wavelet_multiband_importance_per_channel",
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


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def _gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    k1d = _gaussian_kernel1d(float(sigma), device, dtype)
    radius = k1d.numel() // 2
    kx = k1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=C)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, ky, groups=C)
    return x


def _sobel_grad_mag(x: torch.Tensor) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    B, C, H, W = x.shape
    kx = kx.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x_pad, kx, groups=C)
    gy = F.conv2d(x_pad, ky, groups=C)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


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


def wavelet_bandpass_importance_per_channel(
    y: torch.Tensor,
    J: int = 1,
    wave: str = "haar",
    mode: str = "zero",
    beta_w: float = 100.0,
    power: float = 1.5,
    sigma_low: float = 1.5,
    sigma_high: float = 10.0,
    band_sigma: float = 16.0,
    mask_quantile: float = 0.95,
    norm_quantile: float = 0.99,
    normalize_mean: bool = True,
    clip_min: float | None = None,
    clip_max: float | None = 500.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if DWTForward is None:
        raise ImportError(
            "pytorch_wavelets is required for wavelet_bandpass_importance_per_channel "
            "but could not be imported."
        )

    assert y.dim() == 4, "Expected y with shape [B,C,H,W]"
    B, C, H, W = y.shape
    device, dtype = y.device, y.dtype

    dwt = _get_dwt_forward(J, wave, mode, device, dtype)
    _, Yh = dwt(y)

    F_accum = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    for h in Yh:
        F_j = (h * h).sum(dim=2)
        F_j_up = F.interpolate(F_j, size=(H, W), mode="bilinear", align_corners=False)
        F_accum = F_accum + F_j_up

    blur_low = _gaussian_blur2d(F_accum, sigma_low)
    blur_high = _gaussian_blur2d(F_accum, sigma_high)
    E_bp = torch.relu(blur_low - blur_high)

    E_flat = E_bp.view(B, C, -1)
    q = torch.quantile(E_flat, norm_quantile, dim=-1, keepdim=True)
    q_b = q.view(B, C, 1, 1)
    E_norm = torch.clamp(E_bp / (q_b + eps), 0.0, 1.0)

    grad = _sobel_grad_mag(y)
    g_flat = grad.view(B, C, -1)
    gq = torch.quantile(g_flat, mask_quantile, dim=-1, keepdim=True)
    gq_b = gq.view(B, C, 1, 1)
    mask = (grad > gq_b).to(dtype=dtype)
    band = _gaussian_blur2d(mask, band_sigma)
    band_flat = band.view(B, C, -1)
    bmax = band_flat.max(dim=-1, keepdim=True).values
    band = band / (bmax.view(B, C, 1, 1) + eps)
    band = torch.clamp(band, 0.0, 1.0)

    importance = E_norm * band
    weights = 1.0 + (beta_w - 1.0) * (importance ** power)

    if normalize_mean:
        mean = weights.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
        weights = weights / mean

    if clip_min is not None or clip_max is not None:
        weights = torch.clamp(
            weights,
            min=clip_min if clip_min is not None else -float("inf"),
            max=clip_max if clip_max is not None else float("inf"),
        )

    return weights, E_bp


def _normalize_band(x: torch.Tensor, q: float, eps: float) -> torch.Tensor:
    B, C, H, W = x.shape
    flat = x.view(B, C, -1)
    qv = torch.quantile(flat, q, dim=-1, keepdim=True)
    qv = qv.view(B, C, 1, 1)
    return torch.clamp(x / (qv + eps), 0.0, 1.0)


def wavelet_multiband_importance_per_channel(
    y: torch.Tensor,
    J: int = 3,
    wave: str = "haar",
    mode: str = "zero",
    level_weights: list[float] | None = None,
    lowpass_weight: float = 0.2,
    beta_w: float = 100.0,
    power: float = 1.5,
    norm_quantile: float = 0.99,
    normalize_mean: bool = True,
    rescale_max: bool = False,
    clip_min: float | None = None,
    clip_max: float | None = 500.0,
    combine_norm: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if DWTForward is None:
        raise ImportError(
            "pytorch_wavelets is required for wavelet_multiband_importance_per_channel "
            "but could not be imported."
        )

    assert y.dim() == 4, "Expected y with shape [B,C,H,W]"
    B, C, H, W = y.shape
    device, dtype = y.device, y.dtype

    dwt = _get_dwt_forward(J, wave, mode, device, dtype)
    Yl, Yh = dwt(y)

    num_levels = len(Yh)
    if level_weights is None:
        level_weights = [1.0 for _ in range(num_levels)]
    if len(level_weights) < num_levels:
        level_weights = level_weights + [level_weights[-1]] * (num_levels - len(level_weights))
    if len(level_weights) > num_levels:
        level_weights = level_weights[:num_levels]

    bands = []
    weights = []

    for w_j, h in zip(level_weights, Yh):
        if w_j <= 0:
            continue
        F_j = (h * h).sum(dim=2)
        F_j_up = F.interpolate(F_j, size=(H, W), mode="bilinear", align_corners=False)
        F_j_norm = _normalize_band(F_j_up, norm_quantile, eps)
        bands.append(F_j_norm)
        weights.append(float(w_j))

    if lowpass_weight and lowpass_weight > 0:
        F_lp = Yl * Yl
        F_lp_up = F.interpolate(F_lp, size=(H, W), mode="bilinear", align_corners=False)
        F_lp_norm = _normalize_band(F_lp_up, norm_quantile, eps)
        bands.append(F_lp_norm)
        weights.append(float(lowpass_weight))

    if not bands:
        combined = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    else:
        combined = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        for b, w in zip(bands, weights):
            combined = combined + (b * float(w))
        if combine_norm:
            wsum = sum(weights)
            if wsum > 0:
                combined = combined / float(wsum)
        combined = torch.clamp(combined, 0.0, 1.0)

    out = 1.0 + (beta_w - 1.0) * (combined ** power)
    if normalize_mean:
        mean = out.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
        out = out / mean
    if rescale_max:
        maxv = out.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
        out = out / maxv * float(beta_w)
    if clip_min is not None or clip_max is not None:
        out = torch.clamp(
            out,
            min=clip_min if clip_min is not None else -float("inf"),
            max=clip_max if clip_max is not None else float("inf"),
        )
    return out, combined


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
