"""MetricCalculator: unified aggregator for computing time series metrics.

Usage:
    calc = MetricCalculator(task="prediction", mode="point", metrics=["MSE", "MAE"])
    results = calc.compute(target, forecast)
    # -> {"MSE": tensor(0.042), "MAE": tensor(0.159)}

    calc = MetricCalculator(task="generation", mode="probabilistic")
    results = calc.compute_all(real, generated)
"""

import torch

from .metrics.prediction import point as pred_point
from .metrics.prediction import probabilistic as pred_prob
from .metrics.imputation import point as imp_point
from .metrics.imputation import probabilistic as imp_prob
from .metrics.generation import point as gen_point
from .metrics.generation import probabilistic as gen_prob
from .metrics.anomaly import metrics as anom_metrics
from .metrics.classification import metrics as cls_metrics


_METRIC_REGISTRY = {
    ("prediction", "point"): {
        "funcs": pred_point.POINT_METRIC_FUNCS,
        "defaults": pred_point.POINT_METRICS,
    },
    ("prediction", "probabilistic"): {
        "funcs": pred_prob.PROB_METRIC_FUNCS,
        "defaults": pred_prob.PROB_METRICS,
    },
    ("imputation", "point"): {
        "funcs": imp_point.POINT_METRIC_FUNCS,
        "defaults": imp_point.POINT_METRICS,
    },
    ("imputation", "probabilistic"): {
        "funcs": imp_prob.PROB_METRIC_FUNCS,
        "defaults": imp_prob.PROB_METRICS,
    },
    ("generation", "point"): {
        "funcs": gen_point.POINT_METRIC_FUNCS,
        "defaults": gen_point.POINT_METRICS,
    },
    ("generation", "probabilistic"): {
        "funcs": gen_prob.PROB_METRIC_FUNCS,
        "defaults": gen_prob.PROB_METRICS,
    },
    ("anomaly", "default"): {
        "funcs": anom_metrics.METRIC_FUNCS,
        "defaults": anom_metrics.METRICS,
    },
    ("classification", "default"): {
        "funcs": cls_metrics.METRIC_FUNCS,
        "defaults": cls_metrics.METRICS,
    },
}


class MetricCalculator:
    """Unified metric aggregator.

    Args:
        task: "prediction", "imputation", "generation", "anomaly", or "classification".
        mode: "point", "probabilistic", or "default" (for anomaly/classification).
        metrics: list of metric names (case-insensitive). If None, uses all defaults.
    """

    def __init__(self, task, mode="point", metrics=None):
        self.task = task.lower()
        self.mode = mode.lower()
        self._key = (self.task, self.mode)

        if self._key not in _METRIC_REGISTRY:
            raise ValueError(
                f"Unknown task/mode: ({self.task}, {self.mode}). "
                f"Available: {list(_METRIC_REGISTRY.keys())}"
            )

        reg = _METRIC_REGISTRY[self._key]
        self._funcs = reg["funcs"]
        self._defaults = reg["defaults"]

        if metrics is None:
            self._selected = list(self._defaults)
        else:
            self._selected = self._resolve_names(metrics)

    def _resolve_names(self, names):
        """Resolve metric names case-insensitively against available funcs."""
        name_map = {k.lower(): k for k in self._funcs.keys()}
        resolved = []
        for name in names:
            lower = name.lower()
            if lower in name_map:
                resolved.append(name_map[lower])
            else:
                raise ValueError(
                    f"Unknown metric '{name}' for ({self.task}, {self.mode}). "
                    f"Available: {list(self._funcs.keys())}"
                )
        return resolved

    @property
    def available_metrics(self):
        """List all available metrics for this task/mode."""
        return list(self._funcs.keys())

    @property
    def selected_metrics(self):
        """List currently selected metrics."""
        return list(self._selected)

    def compute(self, *args, mask=None, **kwargs):
        """Compute selected metrics.

        For point metrics: compute(target, forecast, mask=None)
        For probabilistic metrics: compute(target, samples, mask=None)
        For generation metrics: compute(real, generated)

        Returns:
            dict mapping metric name -> scalar tensor.
        """
        results = {}
        for name in self._selected:
            fn = self._funcs[name]
            if mask is not None and "mask" in _get_param_names(fn):
                results[name] = fn(*args, mask=mask, **kwargs)
            else:
                results[name] = fn(*args, **kwargs)
        return results

    def compute_all(self, *args, mask=None, **kwargs):
        """Compute all default metrics for this task/mode."""
        prev = self._selected
        self._selected = list(self._defaults)
        results = self.compute(*args, mask=mask, **kwargs)
        self._selected = prev
        return results

    def compute_per_feature(self, target, forecast, mask=None, metrics=None):
        """Compute per-feature metrics. Only for prediction/imputation point mode.

        Returns:
            dict mapping metric name -> (C,) tensor.
        """
        if self.mode != "point" or self.task == "generation":
            raise ValueError("compute_per_feature only supported for prediction/imputation point metrics")

        per_feat_funcs = {}
        if "mse_per_feature" in dir(self._get_module("point")):
            per_feat_funcs["MSE"] = self._get_module("point").mse_per_feature
        if "mae_per_feature" in dir(self._get_module("point")):
            per_feat_funcs["MAE"] = self._get_module("point").mae_per_feature

        if metrics:
            per_feat_funcs = {k: v for k, v in per_feat_funcs.items() if k.lower() in [m.lower() for m in metrics]}

        results = {}
        for name, fn in per_feat_funcs.items():
            if mask is not None:
                results[name] = fn(target, forecast, mask=mask)
            else:
                results[name] = fn(target, forecast)
        return results

    def _get_module(self, mode):
        """Get the point or probabilistic module for current task."""
        from .metrics import prediction, imputation, generation
        task_mod = {"prediction": prediction, "imputation": imputation, "generation": generation}[self.task]
        return getattr(task_mod, mode)

    def __repr__(self):
        return f"MetricCalculator(task='{self.task}', mode='{self.mode}', metrics={self._selected})"


def _get_param_names(fn):
    """Get parameter names of a function."""
    import inspect
    sig = inspect.signature(fn)
    return list(sig.parameters.keys())


def list_available_metrics():
    """List all available metrics organized by task and mode.

    Returns:
        dict of dicts: {task: {mode: [metric_names]}}
    """
    result = {}
    for (task, mode), reg in _METRIC_REGISTRY.items():
        if task not in result:
            result[task] = {}
        result[task][mode] = list(reg["defaults"])
    return result
