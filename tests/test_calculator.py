"""Tests for MetricCalculator."""

import torch
import pytest
import ts_metric as tm


class TestMetricCalculator:
    def setup_method(self):
        torch.manual_seed(42)
        self.target = torch.randn(4, 3, 24)
        self.forecast = self.target + 0.1 * torch.randn(4, 3, 24)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(4, 50, 3, 24)

    def test_prediction_point(self):
        calc = tm.MetricCalculator(task="prediction", mode="point", metrics=["MSE", "MAE"])
        results = calc.compute(self.target, self.forecast)
        assert "MSE" in results
        assert "MAE" in results
        assert results["MSE"].shape == ()

    def test_prediction_probabilistic(self):
        calc = tm.MetricCalculator(task="prediction", mode="probabilistic", metrics=["CRPS", "PICP"])
        results = calc.compute(self.target, self.samples)
        assert "CRPS" in results
        assert "PICP" in results

    def test_compute_all(self):
        calc = tm.MetricCalculator(task="prediction", mode="point")
        results = calc.compute_all(self.target, self.forecast)
        assert len(results) == len(calc.available_metrics)

    def test_case_insensitive(self):
        calc = tm.MetricCalculator(task="prediction", mode="point", metrics=["mse", "MAE"])
        results = calc.compute(self.target, self.forecast)
        assert len(results) == 2

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            tm.MetricCalculator(task="prediction", mode="point", metrics=["nonexistent"])

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            tm.MetricCalculator(task="nonexistent", mode="point")

    def test_imputation_point(self):
        calc = tm.MetricCalculator(task="imputation", mode="point", metrics=["MSE", "MRE"])
        results = calc.compute(self.target, self.forecast)
        assert "MSE" in results
        assert "MRE" in results

    def test_generation(self):
        real = torch.randn(100, 3, 24)
        gen = torch.randn(80, 3, 24)
        calc = tm.MetricCalculator(task="generation", mode="default", metrics=["MDD", "ED"])
        results = calc.compute(real, gen)
        assert "MDD" in results
        assert "ED" in results

    def test_with_mask(self):
        mask = torch.ones(4, 3, 24)
        mask[:, :, 12:] = 0
        calc = tm.MetricCalculator(task="prediction", mode="point", metrics=["MSE"])
        results = calc.compute(self.target, self.forecast, mask=mask)
        assert "MSE" in results

    def test_repr(self):
        calc = tm.MetricCalculator(task="prediction", mode="point", metrics=["MSE"])
        assert "prediction" in repr(calc)

    def test_list_available(self):
        available = tm.list_available_metrics()
        assert "prediction" in available
        assert "point" in available["prediction"]
        assert "MSE" in available["prediction"]["point"]
