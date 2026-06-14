"""Prediction point metrics: MSE, MAE, RMSE, MAPE, sMAPE, ND, R2, correlation.

Input shapes:
  target:   (B, C, T) or (C, T)
  forecast: (B, C, T) or (C, T)
  mask:     optional, broadcastable to (B, C, T). 1=valid, 0=masked.
"""

import torch
from ...utils import _prepare_point, masked_mean, masked_sum, masked_mean_per_feature


def mse(target, forecast, mask=None):
    """Mean Squared Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    return masked_mean((t - f) ** 2, m)


def mae(target, forecast, mask=None):
    """Mean Absolute Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    return masked_mean(torch.abs(t - f), m)


def rmse(target, forecast, mask=None):
    """Root Mean Squared Error."""
    return torch.sqrt(mse(target, forecast, mask))


def mape(target, forecast, mask=None):
    """Mean Absolute Percentage Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = torch.abs(t)
    valid = (denom > 1e-8) & (m > 0.5)
    if valid.sum() < 1:
        return torch.tensor(float('inf'), device=t.device)
    return torch.mean(torch.abs(t - f)[valid] / denom[valid])


def smape(target, forecast, mask=None):
    """Symmetric Mean Absolute Percentage Error."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = torch.abs(t) + torch.abs(f)
    valid = (denom > 1e-8) & (m > 0.5)
    if valid.sum() < 1:
        return torch.tensor(0.0, device=t.device)
    return torch.mean(2.0 * torch.abs(t - f)[valid] / denom[valid])


def nd(target, forecast, mask=None):
    """Normalized Deviation: sum(|t-f|) / sum(|t|)."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = masked_sum(torch.abs(t), m)
    if denom < 1e-8:
        return torch.tensor(float('inf'), device=t.device)
    return masked_sum(torch.abs(t - f), m) / denom


def r2(target, forecast, mask=None):
    """R² (coefficient of determination)."""
    t, f, m = _prepare_point(target, forecast, mask)
    t_masked = t * m
    f_masked = f * m
    t_mean = t_masked.sum() / m.sum().clamp(min=1e-8)
    ss_res = ((t - f) ** 2 * m).sum()
    ss_tot = ((t - t_mean) ** 2 * m).sum()
    if ss_tot < 1e-8:
        return torch.tensor(0.0, device=t.device)
    return 1.0 - ss_res / ss_tot


def correlation(target, forecast, mask=None):
    """Pearson correlation (averaged over batch and features)."""
    t, f, m = _prepare_point(target, forecast, mask)
    B, C, T = t.shape
    corrs = []
    for b in range(B):
        for c in range(C):
            mask_bc = m[b, c]
            valid_idx = mask_bc > 0.5
            if valid_idx.sum() < 2:
                continue
            t_valid = t[b, c][valid_idx]
            f_valid = f[b, c][valid_idx]
            t_m = t_valid - t_valid.mean()
            f_m = f_valid - f_valid.mean()
            cov = (t_m * f_m).sum()
            t_std = torch.sqrt((t_m ** 2).sum())
            f_std = torch.sqrt((f_m ** 2).sum())
            denom = t_std * f_std
            if denom > 1e-8:
                corrs.append(cov / denom)
    if not corrs:
        return torch.tensor(0.0, device=t.device)
    return torch.mean(torch.stack(corrs))


def mse_per_feature(target, forecast, mask=None):
    """Per-feature MSE. Returns (C,)."""
    t, f, m = _prepare_point(target, forecast, mask)
    return masked_mean_per_feature((t - f) ** 2, m)


def mae_per_feature(target, forecast, mask=None):
    """Per-feature MAE. Returns (C,)."""
    t, f, m = _prepare_point(target, forecast, mask)
    return masked_mean_per_feature(torch.abs(t - f), m)


def nrmse(target, forecast, mask=None):
    """Normalized RMSE: RMSE / mean(|y|) (GluonTS: NRMSE)."""
    t, f, m = _prepare_point(target, forecast, mask)
    rmse_val = torch.sqrt(masked_mean((t - f) ** 2, m))
    abs_mean = masked_mean(torch.abs(t), m)
    return rmse_val / abs_mean.clamp(min=1e-8)


POINT_METRICS = ["MSE", "RMSE", "NRMSE", "MAE", "MAPE", "sMAPE", "ND", "R2", "Correlation"]

POINT_METRIC_FUNCS = {
    "MSE": mse,
    "RMSE": rmse,
    "NRMSE": nrmse,
    "MAE": mae,
    "MAPE": mape,
    "sMAPE": smape,
    "ND": nd,
    "R2": r2,
    "Correlation": correlation,
}
