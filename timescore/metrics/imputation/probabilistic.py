"""Imputation probabilistic metrics.

Reuses prediction probabilistic metrics (CRPS, PICP, QICE)
and adds imputation-specific IntervalWidth.

Input shapes:
  target:  (B, C, T) or (C, T)
  samples: (B, S, C, T) or (S, C, T)
  mask:    optional, broadcastable to (B, C, T). 1=valid, 0=masked.
"""

import torch
from ..prediction.probabilistic import crps, picp, qice
from ...utils import _prepare_prob, masked_mean


def interval_width(target, samples, mask=None, alpha=0.1):
    """Average prediction interval width.

    alpha: significance level (e.g., 0.1 for 90% interval).
    """
    t, s, m = _prepare_prob(target, samples, mask)
    q_low = torch.quantile(s, alpha / 2, dim=1)
    q_high = torch.quantile(s, 1 - alpha / 2, dim=1)
    width = q_high - q_low
    return masked_mean(width, m)


PROB_METRICS = ["CRPS", "PICP", "QICE", "IntervalWidth"]

PROB_METRIC_FUNCS = {
    "CRPS": crps,
    "PICP": picp,
    "QICE": qice,
    "IntervalWidth": interval_width,
}
