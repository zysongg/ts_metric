"""timescore: Time series metric computation library.

Supports five tasks: prediction, imputation, generation, anomaly detection, classification.

Quick start:
    import timescore as tm

    # Prediction
    mse = tm.prediction.mse(target, forecast)
    crps = tm.prediction.crps(target, samples)

    # Visualization
    ax = tm.plot_coverage(target, samples)

    # Export
    df = tm.to_dataframe(results)

    # Statistical test
    dm = tm.diebold_mariano(target, forecast_a, forecast_b)
"""

__version__ = "0.3.0"

from . import metrics, utils
from .metrics import prediction, imputation, generation, anomaly, classification
from .calculator import MetricCalculator, list_available_metrics
from .visualization import plot_coverage, plot_quantile_loss, plot_calibration, plot_crps_comparison
from .export import to_dict, to_dataframe, to_json, to_csv
from .statistical import diebold_mariano, paired_t_test
from .per_horizon import per_horizon, per_horizon_prob, horizon_summary

__all__ = [
    "prediction",
    "imputation",
    "generation",
    "anomaly",
    "classification",
    "MetricCalculator",
    "list_available_metrics",
    # Visualization
    "plot_coverage",
    "plot_quantile_loss",
    "plot_calibration",
    "plot_crps_comparison",
    # Export
    "to_dict",
    "to_dataframe",
    "to_json",
    "to_csv",
    # Statistical
    "diebold_mariano",
    "paired_t_test",
    # Per-horizon
    "per_horizon",
    "per_horizon_prob",
    "horizon_summary",
]
