"""Prediction probabilistic metrics: CRPS, PICP, QICE, MSE_median, MAE_median, calibration, log_likelihood.

Input shapes:
  target:  (B, C, T) or (C, T)
  samples: (B, S, C, T) or (S, C, T)
  mask:    optional, broadcastable to (B, C, T). 1=valid, 0=masked.
"""

import torch
from ...utils import _prepare_prob, masked_mean, compute_quantiles_torch, DEFAULT_QUANTILE_LEVELS


def crps(target, samples, mask=None):
    """Continuous Ranked Probability Score (exact formula, O(S^2) per point).

    CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the predictive distribution.

    For S ensemble samples:
    CRPS = (1/S) sum_s |x_s - y| - (1/(2*S^2)) sum_s sum_{s'} |x_s - x_{s'}|
    """
    t, s, m = _prepare_prob(target, samples, mask)
    B, S, C, T = s.shape

    term1 = torch.abs(s - t.unsqueeze(1)).mean(dim=1)

    s_sorted, _ = s.sort(dim=1)
    idx = torch.arange(1, S + 1, device=s.device, dtype=s.dtype).view(1, S, 1, 1)
    weights = (2 * idx - 1 - S) / (S * S)
    term2 = (weights * s_sorted).sum(dim=1)

    crps_val = term1 - term2
    return masked_mean(crps_val, m)


def crps_quantile(target, samples, mask=None):
    """CRPS via quantile loss approximation (O(S log S), uses 7 quantile levels).

    CRPS ≈ 2 * mean_q(quantile_loss(q, target, q_pred))
    """
    t, s, m = _prepare_prob(target, samples, mask)
    quantiles = compute_quantiles_torch(s, DEFAULT_QUANTILE_LEVELS)

    total_loss = torch.tensor(0.0, device=t.device, dtype=t.dtype)
    n_quantiles = 0
    for q, q_pred in quantiles.items():
        diff = t - q_pred
        loss = torch.max(q * diff, (q - 1.0) * diff)
        total_loss = total_loss + masked_mean(loss, m)
        n_quantiles += 1

    return 2.0 * total_loss / max(n_quantiles, 1)


def crps_sum(target, samples, mask=None):
    """CRPS over marginal sums across features."""
    t, s, m = _prepare_prob(target, samples, mask)
    t_sum = t.sum(dim=1, keepdim=True)  # (B, 1, T)
    s_sum = s.sum(dim=2, keepdim=True)  # (B, S, 1, T)
    m_sum = m.sum(dim=1, keepdim=True)  # (B, 1, T)
    return crps(t_sum, s_sum, m_sum)


def picp(target, samples, mask=None, alpha=0.1):
    """Prediction Interval Coverage Probability.

    alpha: significance level (e.g., 0.1 for 90% interval).
    Returns fraction of targets within [alpha/2, 1-alpha/2] quantiles.
    """
    t, s, m = _prepare_prob(target, samples, mask)
    q_low = torch.quantile(s, alpha / 2, dim=1)
    q_high = torch.quantile(s, 1 - alpha / 2, dim=1)
    covered = ((t >= q_low) & (t <= q_high)).float()
    return masked_mean(covered, m)


def qice(target, samples, mask=None):
    """Quantile Interval Coverage Error.

    Average |observed_coverage - expected_coverage| across quantile levels.
    """
    t, s, m = _prepare_prob(target, samples, mask)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    errors = []
    for alpha in levels:
        q_low = torch.quantile(s, alpha / 2, dim=1)
        q_high = torch.quantile(s, 1 - alpha / 2, dim=1)
        covered = ((t >= q_low) & (t <= q_high)).float()
        observed = masked_mean(covered, m)
        errors.append(torch.abs(observed - (1 - alpha)))
    return torch.mean(torch.stack(errors))


def mse_median(target, samples, mask=None):
    """MSE of median forecast vs target."""
    t, s, m = _prepare_prob(target, samples, mask)
    median = torch.quantile(s, 0.5, dim=1)  # (B, C, T)
    return masked_mean((t - median) ** 2, m)


def mae_median(target, samples, mask=None):
    """MAE of median forecast vs target."""
    t, s, m = _prepare_prob(target, samples, mask)
    median = torch.quantile(s, 0.5, dim=1)
    return masked_mean(torch.abs(t - median), m)


def calibration_error(target, samples, mask=None, n_bins=10):
    """Calibration error via reliability diagram.

    Bins predicted quantiles and compares observed vs expected coverage.
    """
    t, s, m = _prepare_prob(target, samples, mask)
    B, C, T = t.shape

    errors = []
    for q_level in torch.linspace(0.05, 0.95, n_bins, device=t.device):
        q_pred = torch.quantile(s, q_level, dim=1)
        observed = masked_mean((t <= q_pred).float(), m)
        errors.append(torch.abs(observed - q_level))

    return torch.mean(torch.stack(errors))


def log_likelihood(target, samples, mask=None):
    """Gaussian log-likelihood score (higher is better).

    Fits per-point Gaussian from samples, returns mean log-likelihood.
    """
    t, s, m = _prepare_prob(target, samples, mask)
    mean = s.mean(dim=1)
    var = s.var(dim=1).clamp(min=1e-6)
    ll = -0.5 * (torch.log(2 * torch.pi * var) + (t - mean) ** 2 / var)
    return masked_mean(ll, m)


PROB_METRICS = ["CRPS", "CRPS_quantile", "CRPS_sum", "PICP", "QICE", "MSE_median", "MAE_median", "Calibration", "LogLikelihood"]

PROB_METRIC_FUNCS = {
    "CRPS": crps,
    "CRPS_quantile": crps_quantile,
    "CRPS_sum": crps_sum,
    "PICP": picp,
    "QICE": qice,
    "MSE_median": mse_median,
    "MAE_median": mae_median,
    "Calibration": calibration_error,
    "LogLikelihood": log_likelihood,
}
