"""Imputation point metrics.

Reuses prediction point metrics (MSE, MAE, RMSE, MAPE, sMAPE, ND)
and adds imputation-specific MRE.

Input shapes:
  target:   (B, C, T) or (C, T)
  forecast: (B, C, T) or (C, T)  -- imputed values
  mask:     optional, broadcastable to (B, C, T). 1=valid, 0=masked.
            For imputation, typically mask=1 at positions where
            ground truth is available for evaluation.
"""

import torch
from ..prediction.point import mse, mae, rmse, mape, smape, nd
from ...utils import _prepare_point, masked_mean


def mre(target, forecast, mask=None):
    """Mean Relative Error: mean(|t-f|) / mean(|t|)."""
    t, f, m = _prepare_point(target, forecast, mask)
    denom = masked_mean(torch.abs(t), m)
    if denom < 1e-8:
        return torch.tensor(float('inf'), device=t.device)
    return masked_mean(torch.abs(t - f), m) / denom


POINT_METRICS = ["MSE", "RMSE", "MAE", "MAPE", "MRE", "sMAPE", "ND"]

POINT_METRIC_FUNCS = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "MAPE": mape,
    "MRE": mre,
    "sMAPE": smape,
    "ND": nd,
}
