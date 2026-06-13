"""Statistical tests for comparing forecast accuracy.

Includes the Diebold-Mariano test for comparing two forecasts.
"""

import torch
import numpy as np


def diebold_mariano(target, forecast_a, forecast_b, loss="mse", h=1):
    """Diebold-Mariano test for comparing two forecasts.

    Tests the null hypothesis that two forecasts have equal predictive accuracy.

    Args:
        target: ground truth, shape (T,) or (B, T)
        forecast_a: first forecast, same shape as target
        forecast_b: second forecast, same shape as target
        loss: loss function, "mse" or "mae"
        h: forecast horizon (for autocorrelation correction)

    Returns:
        dict with keys:
            "statistic": DM test statistic
            "p_value": two-sided p-value
            "significant": bool, True if p < 0.05

    Reference:
        Diebold & Mariano (1995), "Comparing Predictive Accuracy"
    """
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        forecast_a = forecast_a.detach().cpu().numpy()
        forecast_b = forecast_b.detach().cpu().numpy()

    target = target.flatten()
    forecast_a = forecast_a.flatten()
    forecast_b = forecast_b.flatten()

    if loss == "mse":
        loss_a = (target - forecast_a) ** 2
        loss_b = (target - forecast_b) ** 2
    elif loss == "mae":
        loss_a = np.abs(target - forecast_a)
        loss_b = np.abs(target - forecast_b)
    else:
        raise ValueError(f"Unknown loss: {loss}. Use 'mse' or 'mae'.")

    d = loss_a - loss_b
    d_mean = d.mean()

    # Autocovariance up to lag h-1
    T = len(d)
    gamma = np.zeros(h)
    for k in range(h):
        gamma[k] = np.mean((d[k:] - d_mean) * (d[:T - k] - d_mean))

    # Long-run variance
    var_hat = gamma[0] + 2 * np.sum(gamma[1:]) if h > 1 else gamma[0]

    if var_hat < 1e-12:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    dm_stat = d_mean / np.sqrt(var_hat / T)

    # Two-sided p-value using standard normal
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return {
        "statistic": float(dm_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def paired_t_test(metric_a_values, metric_b_values):
    """Paired t-test for comparing metric values across multiple evaluations.

    Args:
        metric_a_values: list or array of metric values from model A
        metric_b_values: list or array of metric values from model B

    Returns:
        dict with keys:
            "statistic": t-test statistic
            "p_value": two-sided p-value
            "significant": bool, True if p < 0.05
    """
    from scipy.stats import ttest_rel

    a = np.asarray(metric_a_values).flatten()
    b = np.asarray(metric_b_values).flatten()

    stat, p_value = ttest_rel(a, b)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }
