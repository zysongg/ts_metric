"""Anomaly detection metrics for time series.

Includes standard binary metrics and time-series specific Point-Adjust F1.

Input shapes:
  labels: (B, T) or (T,) — binary ground truth (1=anomaly, 0=normal)
  scores: (B, T) or (T,) — anomaly scores (higher = more anomalous)
  preds:  (B, T) or (T,) — binary predictions (1=predicted anomaly)
  mask:   optional, broadcastable to labels shape. 1=valid, 0=masked.
"""

import torch
from ...utils import masked_mean


def _ensure_2d(tensor, name="tensor"):
    """Ensure tensor is at least 2D (B, T)."""
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 1D (T,) or 2D (B, T), got {tensor.ndim}D")
    return tensor


def _prepare_anomaly(labels, preds_or_scores, mask=None):
    """Prepare inputs for anomaly detection metrics."""
    labels = _ensure_2d(labels, "labels")
    preds_or_scores = _ensure_2d(preds_or_scores, "preds_or_scores")
    if labels.shape != preds_or_scores.shape:
        raise ValueError(f"Shape mismatch: labels {labels.shape} vs preds/scores {preds_or_scores.shape}")
    if mask is not None:
        mask = _ensure_2d(mask, "mask")
        mask = mask.expand(labels.shape).float()
    else:
        mask = torch.ones_like(labels, dtype=torch.float)
    return labels, preds_or_scores, mask


# --- Standard binary metrics ---

def precision(labels, preds, mask=None):
    """Precision: TP / (TP + FP)."""
    l, p, m = _prepare_anomaly(labels, preds)
    tp = ((p > 0.5) & (l > 0.5) & (m > 0.5)).float().sum()
    fp = ((p > 0.5) & (l < 0.5) & (m > 0.5)).float().sum()
    return tp / (tp + fp).clamp(min=1e-8)


def recall(labels, preds, mask=None):
    """Recall: TP / (TP + FN)."""
    l, p, m = _prepare_anomaly(labels, preds)
    tp = ((p > 0.5) & (l > 0.5) & (m > 0.5)).float().sum()
    fn = ((p < 0.5) & (l > 0.5) & (m > 0.5)).float().sum()
    return tp / (tp + fn).clamp(min=1e-8)


def f1(labels, preds, mask=None):
    """F1 score: harmonic mean of precision and recall."""
    p = precision(labels, preds, mask)
    r = recall(labels, preds, mask)
    return 2 * p * r / (p + r).clamp(min=1e-8)


# --- Time-series specific: Point-Adjust F1 ---

def _point_adjust(labels, preds, mask):
    """Apply point-adjustment: if any point in a contiguous anomaly segment is
    detected, the entire segment is considered detected.

    Args:
        labels: (B, T) binary ground truth
        preds: (B, T) binary predictions
        mask: (B, T) validity mask

    Returns:
        adjusted_preds: (B, T) point-adjusted predictions
    """
    B, T = labels.shape
    adjusted = preds.clone()

    for b in range(B):
        i = 0
        while i < T:
            if labels[b, i] > 0.5 and mask[b, i] > 0.5:
                j = i
                while j < T and labels[b, j] > 0.5 and mask[b, j] > 0.5:
                    j += 1
                segment_preds = preds[b, i:j]
                if (segment_preds > 0.5).any():
                    adjusted[b, i:j] = 1.0
                i = j
            else:
                i += 1

    return adjusted


def pa_precision(labels, preds, mask=None):
    """Point-Adjust Precision."""
    l, p, m = _prepare_anomaly(labels, preds)
    p_adj = _point_adjust(l, p, m)
    tp = ((p_adj > 0.5) & (l > 0.5) & (m > 0.5)).float().sum()
    fp = ((p_adj > 0.5) & (l < 0.5) & (m > 0.5)).float().sum()
    return tp / (tp + fp).clamp(min=1e-8)


def pa_recall(labels, preds, mask=None):
    """Point-Adjust Recall."""
    l, p, m = _prepare_anomaly(labels, preds)
    p_adj = _point_adjust(l, p, m)
    tp = ((p_adj > 0.5) & (l > 0.5) & (m > 0.5)).float().sum()
    fn = ((p_adj < 0.5) & (l > 0.5) & (m > 0.5)).float().sum()
    return tp / (tp + fn).clamp(min=1e-8)


def pa_f1(labels, preds, mask=None):
    """Point-Adjust F1 (PA-F1).

    The standard metric for time series anomaly detection.
    If any point in a contiguous anomaly segment is detected,
    the entire segment counts as detected.
    """
    p = pa_precision(labels, preds, mask)
    r = pa_recall(labels, preds, mask)
    return 2 * p * r / (p + r).clamp(min=1e-8)


# --- AUC metrics ---

def auc_roc(labels, scores, mask=None):
    """Area Under ROC Curve (AUC-ROC).

    Uses trapezoidal rule on sorted thresholds.
    """
    l, s, m = _prepare_anomaly(labels, scores)

    valid = m > 0.5
    l_flat = l[valid]
    s_flat = s[valid]

    if l_flat.sum() == 0 or l_flat.sum() == l_flat.numel():
        return torch.tensor(0.5, device=l.device)

    sorted_idx = s_flat.argsort(descending=True)
    l_sorted = l_flat[sorted_idx]

    n_pos = l_sorted.sum()
    n_neg = l_sorted.numel() - n_pos

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp, fp = 0, 0

    for i in range(l_sorted.numel()):
        if l_sorted[i] > 0.5:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr = torch.tensor(tpr_list, device=l.device, dtype=l.dtype)
    fpr = torch.tensor(fpr_list, device=l.device, dtype=l.dtype)

    auc = torch.trapz(tpr, fpr)
    return auc


def auc_pr(labels, scores, mask=None):
    """Area Under Precision-Recall Curve (AUC-PR)."""
    l, s, m = _prepare_anomaly(labels, scores)

    valid = m > 0.5
    l_flat = l[valid]
    s_flat = s[valid]

    if l_flat.sum() == 0:
        return torch.tensor(0.0, device=l.device)

    sorted_idx = s_flat.argsort(descending=True)
    l_sorted = l_flat[sorted_idx]

    n_pos = l_sorted.sum().item()
    tp, fp = 0, 0

    precision_list = [1.0]
    recall_list = [0.0]

    for i in range(l_sorted.numel()):
        if l_sorted[i] > 0.5:
            tp += 1
        else:
            fp += 1
        precision_list.append(tp / (tp + fp))
        recall_list.append(tp / n_pos)

    prec = torch.tensor(precision_list, device=l.device, dtype=l.dtype)
    rec = torch.tensor(recall_list, device=l.device, dtype=l.dtype)

    auc = torch.trapz(prec, rec)
    return auc


METRICS = ["Precision", "Recall", "F1", "PA_Precision", "PA_Recall", "PA_F1", "AUC_ROC", "AUC_PR"]

METRIC_FUNCS = {
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "PA_Precision": pa_precision,
    "PA_Recall": pa_recall,
    "PA_F1": pa_f1,
    "AUC_ROC": auc_roc,
    "AUC_PR": auc_pr,
}
