"""Per-horizon metric computation.

Compute metrics at each forecast horizon step.
"""

import torch
from typing import Dict, List, Callable, Optional


def per_horizon(metric_fn: Callable, target, forecast, mask=None, horizons=None) -> Dict[int, "torch.Tensor"]:
    """Compute a metric at each forecast horizon step.

    Args:
        metric_fn: metric function with signature fn(target, forecast, mask=None) -> scalar
        target: (B, C, T) or (C, T)
        forecast: same shape as target
        mask: optional mask, same shape as target
        horizons: list of horizon indices to compute (default: all T steps)

    Returns:
        dict mapping horizon index -> metric value tensor
    """
    from ..utils import ensure_3d, clean_tensor, broadcast_mask

    target = ensure_3d(clean_tensor(target, "target"), "target")
    forecast = ensure_3d(clean_tensor(forecast, "forecast"), "forecast")
    T = target.shape[2]

    if horizons is None:
        horizons = list(range(T))

    if mask is not None:
        mask = broadcast_mask(mask, target.shape, device=target.device)

    results = {}
    for h in horizons:
        if h >= T:
            continue
        t_h = target[:, :, h:h + 1]
        f_h = forecast[:, :, h:h + 1]
        m_h = mask[:, :, h:h + 1] if mask is not None else None
        results[h] = metric_fn(t_h, f_h, mask=m_h)

    return results


def per_horizon_prob(metric_fn: Callable, target, samples, mask=None, horizons=None, **kwargs) -> Dict[int, "torch.Tensor"]:
    """Compute a probabilistic metric at each forecast horizon step.

    Args:
        metric_fn: metric function with signature fn(target, samples, mask=None, **kwargs) -> scalar
        target: (B, C, T) or (C, T)
        samples: (B, S, C, T) or (S, C, T)
        mask: optional mask, broadcastable to (B, C, T)
        horizons: list of horizon indices to compute (default: all T steps)
        **kwargs: additional arguments passed to metric_fn

    Returns:
        dict mapping horizon index -> metric value tensor
    """
    from ..utils import ensure_3d, ensure_4d_samples, clean_tensor, broadcast_mask

    target = ensure_3d(clean_tensor(target, "target"), "target")
    samples = ensure_4d_samples(clean_tensor(samples, "samples"), "samples")
    T = target.shape[2]

    if horizons is None:
        horizons = list(range(T))

    if mask is not None:
        mask = broadcast_mask(mask, target.shape, device=target.device)

    results = {}
    for h in horizons:
        if h >= T:
            continue
        t_h = target[:, :, h:h + 1]
        s_h = samples[:, :, :, h:h + 1]
        m_h = mask[:, :, h:h + 1] if mask is not None else None
        results[h] = metric_fn(t_h, s_h, mask=m_h, **kwargs)

    return results


def horizon_summary(results: Dict[int, "torch.Tensor"]) -> "torch.Tensor":
    """Convert per-horizon results to a tensor of shape (T,).

    Args:
        results: dict mapping horizon index -> scalar tensor

    Returns:
        tensor of shape (max_horizon + 1,)
    """
    if not results:
        return torch.tensor([])

    max_h = max(results.keys())
    out = torch.zeros(max_h + 1)
    for h, val in results.items():
        out[h] = val.item() if isinstance(val, torch.Tensor) else val
    return out
