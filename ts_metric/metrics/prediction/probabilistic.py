"""Prediction probabilistic metrics.

Includes GluonTS-compatible metrics (QuantileLoss, wQuantileLoss, Coverage, MSIS)
and additional metrics (CRPS, PICP, QICE, log_likelihood).

Input shapes:
  target:  (B, C, T) or (C, T)
  samples: (B, S, C, T) or (S, C, T)
  mask:    optional, broadcastable to (B, C, T). 1=valid, 0=masked.
"""

import torch
from ...utils import _prepare_prob, masked_mean, masked_sum

GLUONTS_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# --- CRPS ---

def crps_exact(target, samples, mask=None):
    """Exact CRPS: E|X - y| - 0.5 * E|X - X'|.

    Uses the closed-form formula with sorted samples. O(S log S).
    Verified against CRPS.CRPS package (NsDiff reference).
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


def crps(target, samples, mask=None, quantile_levels=None):
    """CRPS as used in DeepAR, K2VAE, GluonTS: mean(wQuantileLoss).

    CRPS = mean_q(QL(q) / sum(|y|))
    """
    return _mean_w_quantile_loss(target, samples, mask, quantile_levels)


def crps_sum_exact(target, samples, mask=None):
    """Exact CRPS computed on marginal sums across features."""
    t, s, m = _prepare_prob(target, samples, mask)
    t_sum = t.sum(dim=1, keepdim=True)
    s_sum = s.sum(dim=2, keepdim=True)
    m_sum = m.sum(dim=1, keepdim=True)
    return crps_exact(t_sum, s_sum, m_sum)


def crps_sum(target, samples, mask=None, quantile_levels=None):
    """CRPS-Sum as used in DeepAR, K2VAE: mean(wQuantileLoss) on feature-summed values."""
    t, s, m = _prepare_prob(target, samples, mask)
    t_sum = t.sum(dim=1, keepdim=True)
    s_sum = s.sum(dim=2, keepdim=True)
    m_sum = m.sum(dim=1, keepdim=True)
    return _mean_w_quantile_loss(t_sum, s_sum, m_sum, quantile_levels)


# --- Quantile-based metrics ---

def _get_quantile_forecast(samples, q):
    """Get quantile forecast from samples. Returns (B, C, T)."""
    return torch.quantile(samples, q, dim=1)


def quantile_loss(target, samples, mask=None, quantile_levels=None):
    """Quantile loss per level (GluonTS-compatible).

    QL(q) = 2 * sum(|(ŷ - y) * (𝟙[y ≤ ŷ] - q)|)

    Returns dict mapping q -> scalar tensor.
    """
    t, s, m = _prepare_prob(target, samples, mask)
    levels = quantile_levels or GLUONTS_QUANTILE_LEVELS
    results = {}
    for q in levels:
        qhat = _get_quantile_forecast(s, q)
        ql = 2.0 * torch.abs((qhat - t) * ((t <= qhat).float() - q))
        results[q] = masked_sum(ql, m)
    return results


def w_quantile_loss(target, samples, mask=None, quantile_levels=None):
    """Weighted quantile loss per level (GluonTS-compatible).

    wQL(q) = QuantileLoss(q) / sum(|y|)

    Returns dict mapping q -> scalar tensor.
    """
    t, s, m = _prepare_prob(target, samples, mask)
    levels = quantile_levels or GLUONTS_QUANTILE_LEVELS
    abs_target_sum = masked_sum(torch.abs(t), m)
    ql = quantile_loss(target, samples, mask, levels)
    return {q: v / abs_target_sum.clamp(min=1e-8) for q, v in ql.items()}


def _mean_w_quantile_loss(target, samples, mask=None, quantile_levels=None):
    """Mean weighted quantile loss across levels (internal, used by crps)."""
    wql = w_quantile_loss(target, samples, mask, quantile_levels)
    return torch.mean(torch.stack(list(wql.values())))


def coverage(target, samples, mask=None, quantile_levels=None):
    """Coverage per quantile level (GluonTS-compatible).

    Coverage(q) = mean(y ≤ ŷ_q)

    Returns dict mapping q -> scalar tensor.
    """
    t, s, m = _prepare_prob(target, samples, mask)
    levels = quantile_levels or GLUONTS_QUANTILE_LEVELS
    results = {}
    for q in levels:
        qhat = _get_quantile_forecast(s, q)
        cov = (t <= qhat).float()
        results[q] = masked_mean(cov, m)
    return results


def mae_coverage(target, samples, mask=None, quantile_levels=None):
    """MAE of coverage across quantile levels (GluonTS: MAE_Coverage).

    MAE_Coverage = mean(|Coverage(q) - q|) over all q.
    """
    levels = quantile_levels or GLUONTS_QUANTILE_LEVELS
    cov = coverage(target, samples, mask, levels)
    errors = [torch.abs(cov[q] - q) for q in levels]
    return torch.mean(torch.stack(errors))


# --- Interval metrics ---

def msis(target, samples, mask=None, alpha=0.05, seasonal_error=None):
    """Mean Scaled Interval Score (GluonTS/M4 competition).

    MSIS = mean(U - L + (2/α)(L-y)𝟙[y<L] + (2/α)(y-U)𝟙[y>U]) / seasonal_error
    """
    t, s, m = _prepare_prob(target, samples, mask)

    lower = _get_quantile_forecast(s, alpha / 2)
    upper = _get_quantile_forecast(s, 1 - alpha / 2)

    width = upper - lower
    under = (2.0 / alpha) * (lower - t) * (t < lower).float()
    over = (2.0 / alpha) * (t - upper) * (t > upper).float()
    score = width + under + over

    if seasonal_error is None:
        seasonal_error = torch.abs(t[:, :, 1:] - t[:, :, :-1]).mean().clamp(min=1e-8)
    else:
        seasonal_error = torch.tensor(seasonal_error, device=t.device, dtype=t.dtype)

    return masked_mean(score, m) / seasonal_error


def picp(target, samples, mask=None, alpha=0.1):
    """Prediction Interval Coverage Probability."""
    t, s, m = _prepare_prob(target, samples, mask)
    q_low = torch.quantile(s, alpha / 2, dim=1)
    q_high = torch.quantile(s, 1 - alpha / 2, dim=1)
    covered = ((t >= q_low) & (t <= q_high)).float()
    return masked_mean(covered, m)


def qice(target, samples, mask=None):
    """Quantile Interval Coverage Error."""
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


# --- Point-from-samples metrics ---

def mse_median(target, samples, mask=None):
    """MSE of median forecast vs target."""
    t, s, m = _prepare_prob(target, samples, mask)
    median = torch.quantile(s, 0.5, dim=1)
    return masked_mean((t - median) ** 2, m)


def mae_median(target, samples, mask=None):
    """MAE of median forecast vs target."""
    t, s, m = _prepare_prob(target, samples, mask)
    median = torch.quantile(s, 0.5, dim=1)
    return masked_mean(torch.abs(t - median), m)


# --- Distributional metrics ---

def log_likelihood(target, samples, mask=None):
    """Gaussian log-likelihood score (higher is better)."""
    t, s, m = _prepare_prob(target, samples, mask)
    mean = s.mean(dim=1)
    var = s.var(dim=1).clamp(min=1e-6)
    ll = -0.5 * (torch.log(2 * torch.pi * var) + (t - mean) ** 2 / var)
    return masked_mean(ll, m)


# --- Multivariate probabilistic metrics ---

def energy_score(target, samples, mask=None):
    """Energy Score: multivariate generalization of CRPS.

    ES = E||X - y|| - 0.5 * E||X - X'||

    where ||.|| is the L2 norm across features and time steps.

    Args:
        target: (B, C, T)
        samples: (B, S, C, T)
        mask: optional mask

    Returns:
        scalar tensor
    """
    t, s, m = _prepare_prob(target, samples, mask)
    B, S, C, T = s.shape

    # Flatten C and T dimensions for norm computation
    t_flat = t.reshape(B, C * T)  # (B, C*T)
    s_flat = s.reshape(B, S, C * T)  # (B, S, C*T)

    # E||X - y||: mean over samples of L2 distance to target
    diff_to_target = torch.norm(s_flat - t_flat.unsqueeze(1), dim=2)  # (B, S)
    term1 = diff_to_target.mean(dim=1)  # (B,)

    # E||X - X'||: mean over all pairs of samples
    # Efficient computation using sorted samples
    s_sorted, _ = s_flat.sort(dim=1)  # (B, S, C*T)
    idx = torch.arange(1, S + 1, device=s.device, dtype=s.dtype).view(1, S, 1)
    weights = (2 * idx - 1 - S) / (S * S)  # (1, S, 1)

    # Use the identity: E||X - X'|| = (2/S^2) * sum_s (2s-1-S) * x_(s) summed over dims
    # But for multivariate, we need pairwise L2 distances
    # Use direct computation for moderate S
    term2_list = []
    for b in range(B):
        pair_dist = torch.cdist(s_flat[b:b+1], s_flat[b:b+1]).squeeze(0)  # (S, S)
        term2_list.append(pair_dist.mean() / 2)
    term2 = torch.stack(term2_list)  # (B,)

    es = term1 - term2  # (B,)
    # Apply mask at batch level
    m_batch = m.reshape(B, -1).mean(dim=1)  # (B,)
    es = es * m_batch
    n = m_batch.sum().clamp(min=1e-8)
    return es.sum() / n


def variogram_score(target, samples, mask=None, p=2):
    """Variogram Score of order p.

    VS_p = mean_{i,j} (|y_i - y_j|^p - E|X_i - X_j|^p)^2

    where i, j are feature-time indices. Computed as mean over all pairs.

    Args:
        target: (B, C, T)
        samples: (B, S, C, T)
        mask: optional mask
        p: order of the variogram (default: 2)

    Returns:
        scalar tensor
    """
    t, s, m = _prepare_prob(target, samples, mask)
    B, S, C, T = s.shape

    t_flat = t.reshape(B, C * T)  # (B, D) where D = C*T
    s_flat = s.reshape(B, S, C * T)  # (B, S, D)
    D = C * T

    # For efficiency, subsample pairs if D is large
    max_pairs = min(D, 50)
    if D > max_pairs:
        idx_i = torch.randint(0, D, (max_pairs,), device=t.device)
        idx_j = torch.randint(0, D, (max_pairs,), device=t.device)
    else:
        idx_i, idx_j = torch.triu_indices(D, D, offset=1, device=t.device)

    # Target pairwise: |t_i - t_j|^p
    t_i = t_flat[:, idx_i]  # (B, P)
    t_j = t_flat[:, idx_j]  # (B, P)
    target_pair = torch.abs(t_i - t_j) ** p  # (B, P)

    # Sample pairwise: E|X_i - X_j|^p
    s_i = s_flat[:, :, idx_i]  # (B, S, P)
    s_j = s_flat[:, :, idx_j]  # (B, S, P)
    sample_pair = (torch.abs(s_i - s_j) ** p).mean(dim=1)  # (B, P)

    vs = (target_pair - sample_pair) ** 2  # (B, P)
    vs = vs.mean(dim=1)  # (B,)

    m_batch = m.reshape(B, -1).mean(dim=1)
    vs = vs * m_batch
    n = m_batch.sum().clamp(min=1e-8)
    return vs.sum() / n


PROB_METRICS = [
    "CRPS", "CRPS_sum", "CRPS_exact", "CRPS_sum_exact",
    "MAE_Coverage", "MSIS",
    "PICP", "QICE",
    "MSE_median", "MAE_median",
    "LogLikelihood",
    "EnergyScore", "VariogramScore",
]

PROB_METRIC_FUNCS = {
    "CRPS": crps,
    "CRPS_sum": crps_sum,
    "CRPS_exact": crps_exact,
    "CRPS_sum_exact": crps_sum_exact,
    "MAE_Coverage": mae_coverage,
    "MSIS": msis,
    "PICP": picp,
    "QICE": qice,
    "MSE_median": mse_median,
    "MAE_median": mae_median,
    "LogLikelihood": log_likelihood,
    "EnergyScore": energy_score,
    "VariogramScore": variogram_score,
}
