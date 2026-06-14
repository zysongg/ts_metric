"""Prediction and imputation visualization with lookback and confidence intervals.

Style reference: TSFLib/tsflib/visual/predictions.py

Input shapes (timescore convention):
  target:      (B, C, T) or (C, T)
  forecast:    (B, C, T) or (C, T)
  samples:     (B, S, C, T) or (S, C, T)
  inputs:      (B, C, L) or (C, L), optional lookback (prediction only)
  mask:        (B, C, T) or broadcastable, for imputation

Optional dependency: matplotlib, scipy (install with `pip install matplotlib scipy`)
"""

import os
from typing import List, Optional, Tuple

import torch
import numpy as np

from ..utils import ensure_3d, ensure_4d_samples


def _check_deps():
    try:
        import matplotlib.pyplot as plt
        import scipy.stats as stats
        return plt, stats
    except ImportError:
        raise ImportError(
            "matplotlib and scipy are required for visualization. "
            "Install with: pip install matplotlib scipy"
        )


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _setup_style():
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ========== Prediction Visualization ==========

def plot_prediction(
    forecast, target, inputs=None,
    sample_id=0, channel=0, lookback_len=48,
    save_path="prediction.png", ax=None,
):
    """Plot single-channel point prediction with optional lookback.

    Args:
        forecast: (B, C, T) or (C, T)
        target:   (B, C, T) or (C, T)
        inputs:   (B, C, L) or (C, L), optional lookback
        sample_id: batch index
        channel: channel index
        lookback_len: number of lookback steps to display
        save_path: output path (ignored if ax is provided)
        ax: matplotlib axes (if None, creates new figure)
    """
    plt, _ = _check_deps()
    _setup_style()

    forecast = _to_numpy(ensure_3d(torch.as_tensor(forecast).float(), "forecast"))
    target = _to_numpy(ensure_3d(torch.as_tensor(target).float(), "target"))

    pred = forecast[sample_id, channel, :]
    true = target[sample_id, channel, :]
    H = len(pred)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    has_lookback = inputs is not None and lookback_len > 0
    if has_lookback:
        inp = _to_numpy(ensure_3d(torch.as_tensor(inputs).float(), "inputs"))
        inp = inp[sample_id, channel, :]
        disp_len = min(lookback_len, len(inp))
        lookback_data = inp[-disp_len:]

        lb_time = np.arange(-disp_len, 0)
        ax.plot(lb_time, lookback_data, color="#1F77B4", linewidth=1.5, label="Lookback")
        ax.plot([-1, 0], [lookback_data[-1], true[0]], color="#1F77B4",
                linewidth=1.5, linestyle=":")
        ax.axvline(x=-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    fc_time = np.arange(H)
    ax.plot(fc_time, true, color="#1F77B4", linewidth=1.5, label="Ground Truth")
    ax.plot(fc_time, pred, color="#D62728", linewidth=1.5, linestyle="--", label="Prediction")
    ax.fill_between(fc_time, np.minimum(pred, true), np.maximum(pred, true),
                    alpha=0.15, color="#D62728", label="Error region")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Prediction — Sample {sample_id}, Channel {channel}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if created_fig:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return ax


def plot_prediction_multi(
    forecast, target, inputs=None,
    sample_id=0, channels=None, lookback_len=48,
    save_path="prediction_multi.png",
):
    """Plot multi-channel point prediction with lookback.

    Args:
        forecast: (B, C, T)
        target:   (B, C, T)
        inputs:   (B, C, L), optional
        sample_id: batch index
        channels: list of channel indices (default: first 4)
        lookback_len: lookback display length
        save_path: output path
    """
    plt, _ = _check_deps()
    _setup_style()

    forecast = _to_numpy(ensure_3d(torch.as_tensor(forecast).float(), "forecast"))
    target = _to_numpy(ensure_3d(torch.as_tensor(target).float(), "target"))
    C = forecast.shape[1]
    if channels is None:
        channels = list(range(min(4, C)))

    n_ch = len(channels)
    H = forecast.shape[2]
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    has_lookback = inputs is not None and lookback_len > 0
    if has_lookback:
        inp = _to_numpy(ensure_3d(torch.as_tensor(inputs).float(), "inputs"))

    for i, ch in enumerate(channels):
        pred = forecast[sample_id, ch, :]
        true = target[sample_id, ch, :]

        if has_lookback:
            inp_ch = inp[sample_id, ch, :]
            disp_len = min(lookback_len, len(inp_ch))
            lookback_data = inp_ch[-disp_len:]
            lb_time = np.arange(-disp_len, 0)
            axes[i].plot(lb_time, lookback_data, color="#1F77B4", linewidth=1.2)
            axes[i].plot([-1, 0], [lookback_data[-1], true[0]], color="#1F77B4",
                         linewidth=1.2, linestyle=":")
            axes[i].axvline(x=-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        fc_time = np.arange(H)
        axes[i].plot(fc_time, true, color="#1F77B4", linewidth=1.2, label="Ground Truth")
        axes[i].plot(fc_time, pred, color="#D62728", linewidth=1.2, linestyle="--", label="Prediction")
        axes[i].set_ylabel(f"Channel {ch}")
        axes[i].grid(True, alpha=0.2)
        axes[i].legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Time Step")
    axes[0].set_title(f"Prediction — Sample {sample_id}, Multi-Channel")
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prob_prediction(
    samples, target, inputs=None,
    sample_id=0, channel=0, lookback_len=48,
    cl=(0.5, 0.9), save_path="prob_prediction.png", ax=None,
):
    """Plot single-channel probabilistic prediction with confidence intervals.

    Uses t-distribution to compute CI from forecast samples.

    Args:
        samples: (B, S, C, T) or (S, C, T)
        target:  (B, C, T) or (C, T)
        inputs:  (B, C, L) or (C, L), optional
        sample_id: batch index
        channel: channel index
        lookback_len: lookback display length
        cl: confidence levels, e.g. (0.5, 0.9)
        save_path: output path
        ax: matplotlib axes
    """
    plt, stats = _check_deps()
    _setup_style()

    samples = _to_numpy(ensure_4d_samples(torch.as_tensor(samples).float(), "samples"))
    target = _to_numpy(ensure_3d(torch.as_tensor(target).float(), "target"))

    y_hats = samples[sample_id, :, channel, :]  # (S, T)
    ys = target[sample_id, channel, :]  # (T,)
    H = len(ys)
    N = y_hats.shape[0]

    y_hat_median = np.median(y_hats, axis=0)
    df = N - 1
    se = stats.sem(y_hats, axis=0)
    intervals = [stats.t.interval(c, df, loc=y_hat_median, scale=se) for c in cl]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    has_lookback = inputs is not None and lookback_len > 0
    if has_lookback:
        inp = _to_numpy(ensure_3d(torch.as_tensor(inputs).float(), "inputs"))
        inp_ch = inp[sample_id, channel, :]
        disp_len = min(lookback_len, len(inp_ch))
        lookback_data = inp_ch[-disp_len:]
        lb_time = np.arange(-disp_len, 0)
        ax.plot(lb_time, lookback_data, color="#1F77B4", linewidth=1.5, label="Lookback")
        ax.plot([-1, 0], [lookback_data[-1], ys[0]], color="#1F77B4",
                linewidth=1.5, linestyle=":")
        ax.axvline(x=-0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    fc_time = np.arange(H)
    colors = ["#61C561", "#339933"]
    alphas = [0.6, 0.85]
    for i in reversed(range(len(cl))):
        ax.fill_between(
            fc_time, intervals[i][0], intervals[i][1],
            alpha=alphas[i], color=colors[i],
            label=f"{cl[i]*100:.0f}% CI", edgecolor="none",
        )

    ax.plot(fc_time, y_hat_median, color="#118811", linewidth=1.5, label="Median")
    ax.plot(fc_time, ys, color="#1F77B4", linewidth=1.5, label="Ground Truth")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title(f"Probabilistic Prediction — Sample {sample_id}, Channel {channel}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if created_fig:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return ax


# ========== Imputation Visualization ==========

def plot_imputation(
    imputed, target, mask=None,
    sample_id=0, channel=0,
    save_path="imputation.png", ax=None,
):
    """Plot single-channel imputation result.

    Shows ground truth, imputed values, and highlights imputed positions.
    No lookback — imputation operates on the full observed series.

    Args:
        imputed:   (B, C, T) or (C, T) — imputed series
        target:    (B, C, T) or (C, T) — ground truth
        mask:      (B, C, T) or broadcastable — 1=observed, 0=missing (imputed)
        sample_id: batch index
        channel: channel index
        save_path: output path
        ax: matplotlib axes
    """
    plt, _ = _check_deps()
    _setup_style()

    imputed = _to_numpy(ensure_3d(torch.as_tensor(imputed).float(), "imputed"))
    target = _to_numpy(ensure_3d(torch.as_tensor(target).float(), "target"))

    imp = imputed[sample_id, channel, :]
    true = target[sample_id, channel, :]
    T_len = len(imp)

    if mask is not None:
        m = _to_numpy(torch.as_tensor(mask).float())
        m = np.broadcast_to(m, target.shape) if m.ndim < target.ndim else m
        m = m[sample_id, channel, :] if m.ndim == 3 else m[channel, :] if m.ndim == 2 else m
        missing = m < 0.5
    else:
        missing = np.zeros(T_len, dtype=bool)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    time_steps = np.arange(T_len)

    # Observed values (where mask=1): solid blue
    observed = ~missing
    if observed.any():
        ax.plot(time_steps[observed], true[observed], "o", color="#1F77B4",
                markersize=2, label="Observed", zorder=4)

    # Ground truth (full line, thin)
    ax.plot(time_steps, true, color="#1F77B4", linewidth=1.0, alpha=0.4, label="Ground Truth")

    # Imputed values at missing positions
    if missing.any():
        ax.plot(time_steps[missing], imp[missing], "o", color="#D62728",
                markersize=4, label="Imputed (missing)", zorder=5)
        for start, end in _find_contiguous(missing):
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.1, color="#D62728")

    # Imputed series (dashed)
    ax.plot(time_steps, imp, color="#D62728", linewidth=1.0, linestyle="--",
            alpha=0.5, label="Imputed series")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    n_missing = int(missing.sum()) if mask is not None else 0
    ax.set_title(f"Imputation — Sample {sample_id}, Channel {channel} ({n_missing} missing)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if created_fig:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return ax


def plot_imputation_multi(
    imputed, target, mask=None,
    sample_id=0, channels=None,
    save_path="imputation_multi.png",
):
    """Plot multi-channel imputation result.

    No lookback — imputation operates on the full observed series.

    Args:
        imputed:   (B, C, T)
        target:    (B, C, T)
        mask:      (B, C, T) or broadcastable
        sample_id: batch index
        channels: list of channel indices (default: first 4)
        save_path: output path
    """
    plt, _ = _check_deps()
    _setup_style()

    imputed = _to_numpy(ensure_3d(torch.as_tensor(imputed).float(), "imputed"))
    target = _to_numpy(ensure_3d(torch.as_tensor(target).float(), "target"))
    C = imputed.shape[1]
    if channels is None:
        channels = list(range(min(4, C)))

    if mask is not None:
        m_full = _to_numpy(torch.as_tensor(mask).float())
        m_full = np.broadcast_to(m_full, target.shape)
    else:
        m_full = None

    n_ch = len(channels)
    T_len = imputed.shape[2]
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    time_steps = np.arange(T_len)

    for i, ch in enumerate(channels):
        imp = imputed[sample_id, ch, :]
        true = target[sample_id, ch, :]
        missing = m_full[sample_id, ch, :] < 0.5 if m_full is not None else np.zeros(T_len, dtype=bool)
        observed = ~missing

        if observed.any():
            axes[i].plot(time_steps[observed], true[observed], "o", color="#1F77B4",
                         markersize=2, zorder=4)

        axes[i].plot(time_steps, true, color="#1F77B4", linewidth=1.0, alpha=0.4, label="Ground Truth")
        axes[i].plot(time_steps, imp, color="#D62728", linewidth=1.0, linestyle="--",
                     alpha=0.5, label="Imputed")

        if missing.any():
            axes[i].plot(time_steps[missing], imp[missing], "o", color="#D62728",
                         markersize=3, label="Imputed (missing)" if i == 0 else None, zorder=5)
            for start, end in _find_contiguous(missing):
                axes[i].axvspan(start - 0.5, end + 0.5, alpha=0.1, color="#D62728")

        axes[i].set_ylabel(f"Channel {ch}")
        axes[i].grid(True, alpha=0.2)
        if i == 0:
            axes[i].legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Time Step")
    axes[0].set_title(f"Imputation — Sample {sample_id}, Multi-Channel")
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prob_imputation(
    samples, target, mask=None,
    sample_id=0, channel=0,
    cl=(0.5, 0.9), save_path="prob_imputation.png", ax=None,
):
    """Plot single-channel probabilistic imputation with confidence intervals.

    No lookback — imputation operates on the full observed series.

    Args:
        samples:   (B, S, C, T) or (S, C, T) — imputed samples
        target:    (B, C, T) or (C, T) — ground truth
        mask:      (B, C, T) or broadcastable — 1=observed, 0=missing
        sample_id: batch index
        channel: channel index
        cl: confidence levels, e.g. (0.5, 0.9)
        save_path: output path
        ax: matplotlib axes
    """
    plt, stats = _check_deps()
    _setup_style()

    samples = _to_numpy(ensure_4d_samples(torch.as_tensor(samples).float(), "samples"))
    target = _to_numpy(ensure_3d(torch.as_tensor(target).float(), "target"))

    y_hats = samples[sample_id, :, channel, :]  # (S, T)
    ys = target[sample_id, channel, :]  # (T,)
    T_len = len(ys)
    N = y_hats.shape[0]

    if mask is not None:
        m = _to_numpy(torch.as_tensor(mask).float())
        m = np.broadcast_to(m, target.shape) if m.ndim < target.ndim else m
        m = m[sample_id, channel, :] if m.ndim == 3 else m[channel, :] if m.ndim == 2 else m
        missing = m < 0.5
    else:
        missing = np.zeros(T_len, dtype=bool)

    y_hat_median = np.median(y_hats, axis=0)
    df = N - 1
    se = stats.sem(y_hats, axis=0)
    intervals = [stats.t.interval(c, df, loc=y_hat_median, scale=se) for c in cl]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    time_steps = np.arange(T_len)

    # Confidence intervals (full range)
    colors = ["#61C561", "#339933"]
    alphas = [0.6, 0.85]
    for i in reversed(range(len(cl))):
        ax.fill_between(
            time_steps, intervals[i][0], intervals[i][1],
            alpha=alphas[i] * 0.4, color=colors[i],
            label=f"{cl[i]*100:.0f}% CI", edgecolor="none",
        )

    # Highlight missing regions with stronger CI shading
    if missing.any():
        for start, end in _find_contiguous(missing):
            for i in reversed(range(len(cl))):
                ax.fill_between(
                    time_steps[start:end+1],
                    intervals[i][0][start:end+1],
                    intervals[i][1][start:end+1],
                    alpha=alphas[i], color=colors[i], edgecolor="none",
                )
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.08, color="#D62728")

    # Observed points
    observed = ~missing
    if observed.any():
        ax.plot(time_steps[observed], ys[observed], "o", color="#1F77B4",
                markersize=2, label="Observed", zorder=5)

    # Ground truth (thin line)
    ax.plot(time_steps, ys, color="#1F77B4", linewidth=1.0, alpha=0.4, label="Ground Truth")

    # Median prediction
    ax.plot(time_steps, y_hat_median, color="#118811", linewidth=1.5, label="Median")

    # Imputed points at missing positions
    if missing.any():
        ax.plot(time_steps[missing], y_hat_median[missing], "o", color="#D62728",
                markersize=4, label="Imputed (missing)", zorder=6)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    n_missing = int(missing.sum()) if mask is not None else 0
    ax.set_title(f"Probabilistic Imputation — Sample {sample_id}, Channel {channel} ({n_missing} missing)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if created_fig:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return ax


def _find_contiguous(mask):
    """Find contiguous True regions in a boolean array. Returns list of (start, end)."""
    if not mask.any():
        return []
    changes = np.diff(mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0]
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask) - 1]])
    return list(zip(starts, ends))
