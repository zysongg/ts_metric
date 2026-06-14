"""Visualization utilities for timescore metrics.

Plotting functions for probabilistic forecast evaluation.

Optional dependency: matplotlib (install with `pip install matplotlib`)
"""

import torch


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def _setup_style():
    """Set Times New Roman font for all plots."""
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


def plot_coverage(target, samples, quantile_levels=None, mask=None, ax=None, **kwargs):
    """Plot observed vs expected coverage for each quantile level.

    Args:
        target: (B, C, T) or (C, T)
        samples: (B, S, C, T) or (S, C, T)
        quantile_levels: list of quantile levels (default: 0.1 to 0.9)
        mask: optional mask
        ax: matplotlib axes (created if None)
        **kwargs: passed to plt.plot

    Returns:
        matplotlib axes
    """
    plt = _check_matplotlib()
    _setup_style()
    from ..metrics.prediction.probabilistic import coverage

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    cov = coverage(target, samples, mask, quantile_levels)
    observed = [cov[q].item() for q in quantile_levels]
    expected = list(quantile_levels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(expected, expected, "--", color="gray", label="Ideal", linewidth=1)
    ax.plot(expected, observed, "o-", label="Observed", **kwargs)
    ax.set_xlabel("Expected Coverage")
    ax.set_ylabel("Observed Coverage")
    ax.set_title("Coverage Plot")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)

    return ax


def plot_quantile_loss(target, samples, quantile_levels=None, mask=None, ax=None, **kwargs):
    """Plot quantile loss per quantile level.

    Args:
        target: (B, C, T) or (C, T)
        samples: (B, S, C, T) or (S, C, T)
        quantile_levels: list of quantile levels
        mask: optional mask
        ax: matplotlib axes
        **kwargs: passed to plt.bar

    Returns:
        matplotlib axes
    """
    plt = _check_matplotlib()
    _setup_style()
    from ..metrics.prediction.probabilistic import w_quantile_loss

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    wql = w_quantile_loss(target, samples, mask, quantile_levels)
    losses = [wql[q].item() for q in quantile_levels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(range(len(quantile_levels)), losses, **kwargs)
    ax.set_xticks(range(len(quantile_levels)))
    ax.set_xticklabels([f"{q:.1f}" for q in quantile_levels])
    ax.set_xlabel("Quantile Level")
    ax.set_ylabel("Weighted Quantile Loss")
    ax.set_title("Quantile Loss per Level")

    return ax


def plot_calibration(target, samples, n_bins=10, mask=None, ax=None, **kwargs):
    """Plot calibration (reliability) diagram.

    Args:
        target: (B, C, T) or (C, T)
        samples: (B, S, C, T) or (S, C, T)
        n_bins: number of bins for the reliability diagram
        mask: optional mask
        ax: matplotlib axes
        **kwargs: passed to plt.bar

    Returns:
        matplotlib axes
    """
    plt = _check_matplotlib()
    _setup_style()
    from ..utils import _prepare_prob, masked_mean

    t, s, m = _prepare_prob(target, samples, mask)

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=t.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed = []
    expected = []
    for i in range(n_bins):
        q_level = bin_centers[i].item()
        q_pred = torch.quantile(s, q_level, dim=1)
        obs = masked_mean((t <= q_pred).float(), m).item()
        observed.append(obs)
        expected.append(q_level)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration", linewidth=1)
    ax.bar(expected, observed, width=1.0 / n_bins * 0.8, alpha=0.7, label="Model", **kwargs)
    ax.set_xlabel("Expected Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration (Reliability) Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def plot_crps_comparison(target, samples_dict, labels=None, ax=None):
    """Compare CRPS across multiple models/samples.

    Args:
        target: (B, C, T) or (C, T)
        samples_dict: dict mapping name -> samples tensor (B, S, C, T)
        labels: optional list of labels (uses dict keys if None)
        ax: matplotlib axes

    Returns:
        matplotlib axes
    """
    plt = _check_matplotlib()
    _setup_style()
    from ..metrics.prediction.probabilistic import crps

    if labels is None:
        labels = list(samples_dict.keys())

    crps_vals = []
    for name in labels:
        crps_vals.append(crps(target, samples_dict[name]).item())

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(range(len(labels)), crps_vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("CRPS")
    ax.set_title("CRPS Comparison")

    return ax
