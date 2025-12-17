# rs_metrics_losses.py
# Losses and metrics for phase-field surrogates.
# No second-person phrasing in comments.

from __future__ import annotations
from typing import Dict, Tuple, Optional, Sequence
import math
import torch
import torch.nn.functional as F

try:
    # Optional PhysicsNeMo MSE; falls back to torch.mean((.)**2)
    from physicsnemo.metrics.general.mse import mse as pn_mse  # type: ignore
except Exception:
    pn_mse = None


# -------------------- finite-difference helpers --------------------

def _diff_x(img: torch.Tensor) -> torch.Tensor:
    # Central differences with replicate padding; assumes shape (..., H, W)
    gx = F.pad(img, (1,1,0,0), mode="replicate")
    return 0.5 * (gx[..., :, 2:] - gx[..., :, :-2])

def _diff_y(img: torch.Tensor) -> torch.Tensor:
    gy = F.pad(img, (0,0,1,1), mode="replicate")
    return 0.5 * (gy[..., 2:, :] - gy[..., :-2, :])


# -------------------- interface band and geometry --------------------

def interface_band(phi: torch.Tensor,
                   level: float = 0.5,
                   eps: float = 0.02) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns mask indicating an |phi-level|<=eps band and the gradients gx, gy.
    Args:
      phi: (..., H, W) in [0,1].
      level: interface level set.
      eps: half-width of band for stable geometry evaluation.
    """
    gx = _diff_x(phi)
    gy = _diff_y(phi)
    band = (phi[..., 1:-1, 1:-1] - level).abs() <= eps  # interior consistent with grads
    return band, gx, gy


def curvature_levelset(phi: torch.Tensor,
                       level: float = 0.5,
                       eps_band: float = 0.02,
                       eps_div: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes level-set curvature κ = div(∇φ/||∇φ||) on an interface band.
    Returns (kappa_full, mask) shaped (..., H-2, W-2) to match gradient domain.
    """
    band, gx, gy = interface_band(phi, level=level, eps=eps_band)
    gnorm = (gx**2 + gy**2).add_(eps_div).sqrt()
    nx = gx / gnorm
    ny = gy / gnorm
    kx = _diff_x(nx)
    ky = _diff_y(ny)
    kappa = kx + ky  # divergence
    kappa = kappa[..., 1:-1, 1:-1]  # align with band
    return kappa, band


def interface_perimeter(phi: torch.Tensor,
                        level: float = 0.5,
                        eps_delta: float = 0.02,
                        pixel_size: Tuple[float, float] = (1.0, 1.0)) -> torch.Tensor:
    """
    Approximates perimeter p ≈ ∬ |∇φ| δ_ε(φ-level) dx dy with δ_ε(z)=ε/(π(ε^2+z^2)).
    Returns scalar tensor.
    """
    Hs, Ws = pixel_size
    gx = _diff_x(phi)
    gy = _diff_y(phi)
    gradn = (gx**2 + gy**2).sqrt()
    z = phi[..., 1:-1, 1:-1] - level
    delta = (eps_delta / math.pi) / (eps_delta**2 + z**2)
    length_density = gradn[..., 1:-1, 1:-1] * delta
    area = Hs * Ws
    return length_density.sum() * area


def edge_gradient_strength(phi: torch.Tensor,
                           level: float = 0.5,
                           eps_band: float = 0.02) -> torch.Tensor:
    """
    Mean |∇φ| on the interface band. Useful as an edge sharpness indicator.
    """
    band, gx, gy = interface_band(phi, level=level, eps=eps_band)
    gmag = (gx**2 + gy**2).sqrt()[..., 1:-1, 1:-1]
    num = gmag[band].mean() if band.any() else gmag.mean()
    return num


# -------------------- ligament height and penetration --------------------

def _first_crossing_y_from_top(phi: torch.Tensor,
                               level: float = 0.5) -> torch.Tensor:
    """
    For each x, finds the first y (from top) where phi crosses 'level' by linear interp.
    Returns y_front in pixel coordinates, shape (..., W-2). If no crossing, returns NaN.
    """
    # Work on interior to avoid boundary issues with diffs
    img = phi[..., 1:-1, 1:-1]
    H, W = img.shape[-2:]
    # Boolean where below and above
    below = img < level
    # Crossing between consecutive y where sign changes: below(y) != below(y+1)
    cross = below[..., :-1, :] ^ below[..., 1:, :]
    # Build indices of first crossing along y
    # If none, mark NaN
    y_idx = cross.float().argmax(dim=-2)  # first index along y (0..H-2)
    has = cross.any(dim=-2)
    # Linear interpolation: y + (level - v0)/(v1-v0)
    y0 = y_idx.clamp_max(H-2)
    v0 = img.gather(-2, y0.unsqueeze(-2)).squeeze(-2)
    v1 = img.gather(-2, (y0+1).unsqueeze(-2)).squeeze(-2)
    denom = (v1 - v0).clamp(min=1e-6, max=None)
    alpha = (level - v0) / denom
    y_front = y0.to(img.dtype) + alpha
    y_front = torch.where(has, y_front, torch.full_like(y_front, float('nan')))
    return y_front  # pixel units in [0, H-2]


def _extrema_1d(yx: torch.Tensor, min_prom: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds local minima and maxima along x by sign of first difference.
    NaNs ignored. Returns (mins, maxs) values.
    """
    y = yx.clone()
    # Mask NaNs
    m = torch.isnan(y)
    if m.any():
        # simple fill to avoid propagating NaNs in diffs
        y[m] = torch.nanmean(y)
    dy = F.pad(y.unsqueeze(-2), (1,1,0,0), mode="replicate").squeeze(-2)
    dy = 0.5 * (dy[..., 2:] - dy[..., :-2])  # central diff
    sgn = dy.sign()
    sgn_change = F.pad(sgn.unsqueeze(-2), (1,1,0,0), mode="replicate").squeeze(-2)
    prev = sgn_change[..., :-1]
    nxt  = sgn_change[..., 1:]
    is_max = (prev > 0) & (nxt < 0)
    is_min = (prev < 0) & (nxt > 0)
    ymax = y[..., 1:-1][is_max]
    ymin = y[..., 1:-1][is_min]
    # Optional prominence filter (very light)
    if ymax.numel() and ymin.numel():
        ymean = torch.nanmean(y)
        ymax = ymax[torch.abs(ymax - ymean) >= min_prom] if ymax.numel() else ymax
        ymin = ymin[torch.abs(ymin - ymean) >= min_prom] if ymin.numel() else ymin
    return ymin, ymax


def ligament_height_and_penetration(phi: torch.Tensor,
                                    level: float = 0.5,
                                    y_top_is_0: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximates mean ligament height and maximum penetration depth from a single-valued
    y_front(x) obtained by first crossing along y from the top boundary.

    Returns:
      mean_height: E[Smax] - E[Smin] in pixel units (see Eq. (19) concept).
      max_penetration: normalised depth in [0,1] from top boundary.

    Notes:
      If multiple crossings occur, only the first front is used. For datasets where the
      interface becomes multi-valued in y(x), this is a lower-order approximation.
    """
    y_front = _first_crossing_y_from_top(phi, level=level)  # (..., W-2)
    H = phi.shape[-2] - 2  # interior height
    ymin, ymax = _extrema_1d(y_front, min_prom=0.0)
    if ymin.numel() == 0 or ymax.numel() == 0:
        mean_height = torch.tensor(float('nan'), dtype=phi.dtype, device=phi.device)
    else:
        mean_height = torch.nanmean(ymax) - torch.nanmean(ymin)

    if y_top_is_0:
        # Depth increases downward. Normalise by domain height.
        # max penetration uses the deepest local minimum along the front.
        if ymin.numel():
            max_pen = torch.nanmax(ymin) / max(H, 1)
        else:
            max_pen = torch.tensor(float('nan'), dtype=phi.dtype, device=phi.device)
    else:
        # If origin at bottom, invert.
        if ymin.numel():
            max_pen = 1.0 - (torch.nanmin(ymin) / max(H, 1))
        else:
            max_pen = torch.tensor(float('nan'), dtype=phi.dtype, device=phi.device)

    return mean_height, max_pen


# -------------------- masses and conservation --------------------

def total_mass(u: torch.Tensor,
               pixel_size: Tuple[float, float] = (1.0, 1.0)) -> torch.Tensor:
    """
    Discrete integral over Ω: sum(u)*dx*dy. Supports batched or single field.
    """
    Hs, Ws = pixel_size
    area = Hs * Ws
    return u.sum(dim=(-2, -1)) * area  # preserves leading dims


def relative_mass_error(u_pred: torch.Tensor,
                        u_true: torch.Tensor,
                        pixel_size: Tuple[float, float] = (1.0, 1.0),
                        eps: float = 1e-12) -> torch.Tensor:
    """
    ||m_pred - m_true|| / ||m_true|| with per-sample reduction to scalar.
    """
    mp = total_mass(u_pred, pixel_size=pixel_size)
    mt = total_mass(u_true, pixel_size=pixel_size)
    return (mp - mt).abs() / (mt.abs() + eps)


# -------------------- auto-correlation (Eq. (1) relative error) --------------------

def _fft_autocorr_2d(u: torch.Tensor) -> torch.Tensor:
    """
    Returns circular auto-correlation via FFT: ifft2(|fft2(u_c)|^2), mean-subtracted and normalised.
    Shape preserved. No radial averaging here.
    """
    uc = u - u.mean(dim=(-2, -1), keepdim=True)
    F2 = torch.fft.rfft2(uc)
    ac = torch.fft.irfft2((F2.conj() * F2), s=uc.shape[-2:])
    # Normalise by variance * Npix to get correlation-like scale in [−1,1]
    var = uc.var(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    acn = ac / (var * u.shape[-2] * u.shape[-1])
    return acn


def _radial_average(map2d: torch.Tensor) -> torch.Tensor:
    """
    Fast radial average using precomputed bins for a given HxW; builds bins on the fly.
    """
    H, W = map2d.shape[-2:]
    # Build radius grid centred at (0,0) for circular correlation (FFT convention).
    yy = torch.fft.fftfreq(H, d=1.0).to(map2d.device) * H
    xx = torch.fft.rfftfreq(W, d=1.0).to(map2d.device) * W
    # Switch back to spatial ordering: use absolute distances in pixel units
    # Here, approximate by Euclidean radii on integer grid
    y = torch.arange(H, device=map2d.device) - H//2
    x = torch.arange(W, device=map2d.device) - W//2
    Y, X = torch.meshgrid(y, x, indexing="ij")
    R = torch.sqrt((Y.float())**2 + (X.float())**2)
    r = R.view(-1)
    v = map2d.view(-1)
    rmax = int(R.max().item())
    rb = torch.arange(rmax+1, device=map2d.device)
    # Bin means
    # To avoid scatter_add on very large tensors, compute via grouping
    # Build mask per radius integer bucket
    # Note: for speed-critical paths, precompute these masks once for fixed H,W.
    means = []
    for k in range(rmax+1):
        m = (r >= k) & (r < k+1)
        if m.any():
            means.append(v[m].mean())
        else:
            means.append(torch.tensor(float('nan'), device=map2d.device, dtype=map2d.dtype))
    return torch.stack(means, 0)


def autocorr_rel_error(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    """
    Implements e_AC(u, û, t) = ||S_{ûû}(r) - S_{uu}(r)||_2 / ||S_{uu}(r)||_2, Eq. (1).
    Accepts shapes: (B, H, W) or (B, C, H, W); averages over channels if present.
    Returns per-sample scalar; if channels present, mean across channels.
    """
    if u_pred.dim() == 4:
        B, C, H, W = u_pred.shape
        errs = []
        for c in range(C):
            ac_p = _fft_autocorr_2d(u_pred[:, c])
            ac_t = _fft_autocorr_2d(u_true[:, c])
            rp = torch.stack([_radial_average(ac_p[b]) for b in range(B)], 0)  # (B, R)
            rt = torch.stack([_radial_average(ac_t[b]) for b in range(B)], 0)
            num = torch.nan_to_num((rp - rt), nan=0.0).pow(2).sum(dim=1).sqrt()
            den = torch.nan_to_num(rt, nan=0.0).pow(2).sum(dim=1).sqrt().clamp_min(1e-12)
            errs.append(num / den)
        return torch.stack(errs, 0).mean(dim=0)  # (B,)
    elif u_pred.dim() == 3:
        B, H, W = u_pred.shape
        ac_p = _fft_autocorr_2d(u_pred)
        ac_t = _fft_autocorr_2d(u_true)
        rp = torch.stack([_radial_average(ac_p[b]) for b in range(B)], 0)
        rt = torch.stack([_radial_average(ac_t[b]) for b in range(B)], 0)
        num = torch.nan_to_num((rp - rt), nan=0.0).pow(2).sum(dim=1).sqrt()
        den = torch.nan_to_num(rt, nan=0.0).pow(2).sum(dim=1).sqrt().clamp_min(1e-12)
        return num / den
    else:
        raise ValueError("Expected (B,H,W) or (B,C,H,W).")


# -------------------- field loss and composite loss --------------------

def mse_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if pn_mse is not None:
        return pn_mse(yhat, y)
    return torch.mean((yhat - y) ** 2)


def composite_loss(yhat: torch.Tensor,
                   y: torch.Tensor,
                   *,
                   phi_channel: Optional[int] = None,
                   mass_channels: Optional[Sequence[int]] = None,
                   weights: Dict[str, float] = None,
                   geom_cfg: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Weighted sum of losses. Returns (loss, scalars_dict).
    weights keys:
      - mse, mass_rel, ac, curv_mean_rel, perim_rel, edge_strength_rel
    geom_cfg keys:
      - level, eps_band, eps_delta, pixel_h, pixel_w
    """
    if weights is None: weights = {}
    if geom_cfg is None: geom_cfg = {}
    level = float(geom_cfg.get("level", 0.5))
    eps_band = float(geom_cfg.get("eps_band", 0.02))
    eps_delta = float(geom_cfg.get("eps_delta", 0.02))
    px = float(geom_cfg.get("pixel_h", 1.0))
    py = float(geom_cfg.get("pixel_w", 1.0))

    out: Dict[str, float] = {}
    total = torch.zeros((), dtype=y.dtype, device=y.device)

    # Base MSE
    l_mse = mse_loss(yhat, y)
    total = total + weights.get("mse", 1.0) * l_mse
    out["loss/mse"] = float(l_mse.detach().cpu())

    # Mass conservation on selected channels (e.g., φ, c_A, c_B)
    if mass_channels is not None and len(mass_channels) > 0:
        rels = []
        for c in mass_channels:
            rel = relative_mass_error(yhat[:, c], y[:, c], pixel_size=(px, py)).mean()
            rels.append(rel)
        l_mass = torch.stack(rels).mean()
        total = total + weights.get("mass_rel", 0.0) * l_mass
        out["loss/mass_rel"] = float(l_mass.detach().cpu())

    # Auto-correlation relative error (across all output channels)
    if weights.get("ac", 0.0) > 0.0:
        l_ac = autocorr_rel_error(yhat, y).mean()
        total = total + weights.get("ac", 0.0) * l_ac
        out["loss/ac"] = float(l_ac.detach().cpu())

    # Geometry on φ channel
    if phi_channel is not None:
        ph_pred = yhat[:, phi_channel]
        ph_true = y[:,  phi_channel]

        # Curvature mean relative error using level-set band statistics
        k_pred, m_pred = curvature_levelset(ph_pred, level=level, eps_band=eps_band)
        k_true, m_true = curvature_levelset(ph_true, level=level, eps_band=eps_band)
        mu_k_pred = k_pred.abs()[m_pred].mean() if m_pred.any() else k_pred.abs().mean()
        mu_k_true = k_true.abs()[m_true].mean() if m_true.any() else k_true.abs().mean()
        rel_curv_mean = (mu_k_pred - mu_k_true).abs() / (mu_k_true.abs().clamp_min(1e-12))
        if weights.get("curv_mean_rel", 0.0) > 0.0:
            total = total + weights.get("curv_mean_rel", 0.0) * rel_curv_mean
        out["loss/curv_mean_rel"] = float(rel_curv_mean.detach().cpu())

        # Perimeter relative error
        p_pred = interface_perimeter(ph_pred, level=level, eps_delta=eps_delta, pixel_size=(px, py))
        p_true = interface_perimeter(ph_true, level=level, eps_delta=eps_delta, pixel_size=(px, py))
        rel_perim = (p_pred - p_true).abs() / (p_true.abs().clamp_min(1e-12))
        if weights.get("perim_rel", 0.0) > 0.0:
            total = total + weights.get("perim_rel", 0.0) * rel_perim
        out["loss/perim_rel"] = float(rel_perim.detach().cpu())

        # Edge gradient strength match (relative error on band-mean |∇φ|)
        es_pred = edge_gradient_strength(ph_pred, level=level, eps_band=eps_band)
        es_true = edge_gradient_strength(ph_true, level=level, eps_band=eps_band)
        es_rel = (es_pred - es_true).abs() / (es_true.abs().clamp_min(1e-12))
        if weights.get("edge_strength_rel", 0.0) > 0.0:
            total = total + weights.get("edge_strength_rel", 0.0) * es_rel
        out["loss/edge_strength_rel"] = float(es_rel.detach().cpu())

        # Ligament height and penetration depth metrics (logged only; usually not trained on)
        with torch.no_grad():
            h_pred, pen_pred = ligament_height_and_penetration(ph_pred, level=level, y_top_is_0=True)
            h_true, pen_true = ligament_height_and_penetration(ph_true, level=level, y_top_is_0=True)
            out["metric/ligament_height_rel"] = float(((h_pred - h_true).abs() / (h_true.abs().clamp_min(1e-12))).detach().cpu())
            out["metric/penetration_rel"] = float(((pen_pred - pen_true).abs() / (pen_true.abs().clamp_min(1e-12))).detach().cpu())

    return total, out
