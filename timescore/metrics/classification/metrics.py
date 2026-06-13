"""Classification metrics for time series.

Standard classification metrics for time series classification tasks.

Input shapes:
  labels: (N,) — integer class labels
  preds:  (N,) — predicted class labels
  scores: (N, C) — predicted class probabilities (C = num classes)
"""

import torch


def _validate_labels(labels, preds_or_scores):
    """Validate and flatten classification inputs."""
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D (N,), got {labels.ndim}D")
    if preds_or_scores.ndim < 1 or preds_or_scores.ndim > 2:
        raise ValueError(f"preds/scores must be 1D (N,) or 2D (N, C), got {preds_or_scores.ndim}D")
    if labels.shape[0] != preds_or_scores.shape[0]:
        raise ValueError(f"Batch mismatch: labels {labels.shape[0]}, preds/scores {preds_or_scores.shape[0]}")
    return labels.long(), preds_or_scores


def accuracy(labels, preds):
    """Classification accuracy."""
    l, p = _validate_labels(labels, preds)
    if p.ndim == 2:
        p = p.argmax(dim=1)
    return (l == p).float().mean()


def precision(labels, preds, average="macro"):
    """Precision score.

    average: 'macro' (mean per-class), 'micro' (global TP/FP).
    """
    l, p = _validate_labels(labels, preds)
    if p.ndim == 2:
        p = p.argmax(dim=1)

    classes = torch.unique(l)

    if average == "micro":
        tp = (l == p).float().sum()
        return tp / l.numel()

    precisions = []
    for c in classes:
        tp = ((p == c) & (l == c)).float().sum()
        fp = ((p == c) & (l != c)).float().sum()
        prec = tp / (tp + fp).clamp(min=1e-8)
        precisions.append(prec)

    return torch.mean(torch.stack(precisions))


def recall(labels, preds, average="macro"):
    """Recall score.

    average: 'macro' (mean per-class), 'micro' (global TP/FN).
    """
    l, p = _validate_labels(labels, preds)
    if p.ndim == 2:
        p = p.argmax(dim=1)

    classes = torch.unique(l)

    if average == "micro":
        tp = (l == p).float().sum()
        return tp / l.numel()

    recalls = []
    for c in classes:
        tp = ((p == c) & (l == c)).float().sum()
        fn = ((p != c) & (l == c)).float().sum()
        rec = tp / (tp + fn).clamp(min=1e-8)
        recalls.append(rec)

    return torch.mean(torch.stack(recalls))


def f1(labels, preds, average="macro"):
    """F1 score.

    average: 'macro' (mean per-class F1), 'micro' (global precision/recall).
    """
    l, p = _validate_labels(labels, preds)
    if p.ndim == 2:
        p = p.argmax(dim=1)

    if average == "micro":
        acc = accuracy(l, p)
        return acc

    classes = torch.unique(l)
    f1s = []
    for c in classes:
        tp = ((p == c) & (l == c)).float().sum()
        fp = ((p == c) & (l != c)).float().sum()
        fn = ((p != c) & (l == c)).float().sum()
        prec = tp / (tp + fp).clamp(min=1e-8)
        rec = tp / (tp + fn).clamp(min=1e-8)
        f1_c = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
        f1s.append(f1_c)

    return torch.mean(torch.stack(f1s))


def auc_roc(labels, scores):
    """AUC-ROC for binary classification.

    labels: (N,) binary (0/1)
    scores: (N,) or (N, 2) — if 2D, uses column 1 as positive class score.
    """
    l, s = _validate_labels(labels, scores)

    if s.ndim == 2:
        s = s[:, 1] if s.shape[1] == 2 else s[:, -1]

    l = l.float()
    if l.sum() == 0 or l.sum() == l.numel():
        return torch.tensor(0.5, device=l.device)

    sorted_idx = s.argsort(descending=True)
    l_sorted = l[sorted_idx]

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

    return torch.trapz(tpr, fpr)


METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC_ROC"]

METRIC_FUNCS = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "AUC_ROC": auc_roc,
}
