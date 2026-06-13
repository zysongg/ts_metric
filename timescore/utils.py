"""Shared utilities: shape validation, mask broadcasting, NaN/inf cleaning."""

import warnings

import torch


def ensure_3d(tensor, name="tensor"):
    """Ensure shape is (B, C, T). Accepts (C, T) -> (1, C, T)."""
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be 2D (C, T) or 3D (B, C, T), got {tensor.ndim}D shape {tensor.shape}")
    return tensor


def ensure_4d_samples(tensor, name="samples"):
    """Ensure shape is (B, S, C, T). Accepts (S, C, T) -> (1, S, C, T)."""
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be 3D (S, C, T) or 4D (B, S, C, T), got {tensor.ndim}D shape {tensor.shape}")
    return tensor


def broadcast_mask(mask, target_shape, device=None):
    """Broadcast mask to target_shape (B, C, T).

    Accepts: None (all valid), (T,), (C, T), (B, T), (B, 1, T), (B, C, T).
    Returns float tensor of shape target_shape with 1=valid, 0=masked.
    """
    if mask is None:
        return torch.ones(target_shape, device=device)

    mask = mask.float()

    if mask.ndim == 1:
        # (T,) -> (1, 1, T)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 2:
        if mask.shape[0] == target_shape[1] and mask.shape[1] == target_shape[2]:
            # (C, T) -> (1, C, T)
            mask = mask.unsqueeze(0)
        elif mask.shape[0] == target_shape[0] and mask.shape[1] == target_shape[2]:
            # (B, T) -> (B, 1, T)
            mask = mask.unsqueeze(1)
        else:
            raise ValueError(f"Cannot broadcast 2D mask {mask.shape} to target {target_shape}")
    elif mask.ndim == 3:
        pass
    else:
        raise ValueError(f"mask must be 1D-3D, got {mask.ndim}D")

    return mask.expand(target_shape)


def clean_tensor(tensor, name="tensor"):
    """Warn and replace NaN/inf with 0."""
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        warnings.warn(f"{name} contains {nan_count} NaN and {inf_count} inf, replaced with 0")
        tensor = torch.where(
            torch.isfinite(tensor), tensor,
            torch.zeros(1, device=tensor.device, dtype=tensor.dtype)
        )
    return tensor


def masked_mean(values, mask):
    """Compute mean of values where mask=1. Returns scalar tensor."""
    values = values * mask
    n = mask.sum()
    if n < 1e-8:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)
    return values.sum() / n


def masked_sum(values, mask):
    """Compute sum of values where mask=1. Returns scalar tensor."""
    return (values * mask).sum()


def masked_mean_per_feature(values, mask):
    """Compute mean over (B, T) for each feature. Returns (C,)."""
    values = values * mask
    n = mask.sum(dim=(0, 2))  # (C,)
    n = n.clamp(min=1e-8)
    return values.sum(dim=(0, 2)) / n


def _prepare_point(target, forecast, mask=None):
    """Validate and prepare inputs for point metrics. Returns (target, forecast, mask) all (B, C, T)."""
    target = ensure_3d(clean_tensor(target, "target"), "target")
    forecast = ensure_3d(clean_tensor(forecast, "forecast"), "forecast")
    if target.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: target {target.shape} vs forecast {forecast.shape}")
    mask = broadcast_mask(mask, target.shape, device=target.device)
    return target, forecast, mask


def _prepare_prob(target, samples, mask=None):
    """Validate and prepare inputs for probabilistic metrics.
    Returns (target, samples, mask) with shapes (B,C,T), (B,S,C,T), (B,C,T).
    """
    target = ensure_3d(clean_tensor(target, "target"), "target")
    samples = ensure_4d_samples(clean_tensor(samples, "samples"), "samples")
    B, C, T = target.shape
    if samples.shape[0] != B:
        raise ValueError(f"Batch mismatch: target B={B}, samples B={samples.shape[0]}")
    if samples.shape[2] != C or samples.shape[3] != T:
        raise ValueError(
            f"Shape mismatch: target (B={B}, C={C}, T={T}), "
            f"samples (B={samples.shape[0]}, S={samples.shape[1]}, C={samples.shape[2]}, T={samples.shape[3]})"
        )
    mask = broadcast_mask(mask, target.shape, device=target.device)
    return target, samples, mask


DEFAULT_QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def compute_quantiles_torch(samples, quantile_levels):
    """Compute quantiles from probabilistic samples tensor (B, S, C, T).

    Returns dict mapping quantile level -> tensor of shape (B, C, T).
    """
    result = {}
    for q in quantile_levels:
        result[q] = torch.quantile(samples, q, dim=1)
    return result
