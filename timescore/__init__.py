"""timescore: Time series metric computation library.

Supports five tasks: prediction, imputation, generation, anomaly detection, classification.

Quick start:
    import timescore as tm

    # Prediction
    mse = tm.prediction.mse(target, forecast)
    crps = tm.prediction.crps(target, samples)

    # Anomaly Detection
    pa_f1 = tm.anomaly.pa_f1(labels, preds)

    # Classification
    acc = tm.classification.accuracy(labels, preds)
"""

__version__ = "0.2.0"

from . import metrics, utils
from .metrics import prediction, imputation, generation, anomaly, classification
from .calculator import MetricCalculator, list_available_metrics

__all__ = [
    "prediction",
    "imputation",
    "generation",
    "anomaly",
    "classification",
    "MetricCalculator",
    "list_available_metrics",
]
