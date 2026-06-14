"""Tests for high-priority features: export, statistical, per-horizon, multivariate."""

import torch
import numpy as np
import json
import pytest
import ts_metric as tm


class TestExport:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)
        self.calc = tm.MetricCalculator(task="prediction", mode="point", metrics=["MSE", "MAE", "RMSE"])
        self.results = self.calc.compute(self.target, self.forecast)

    def test_to_dict(self):
        d = tm.to_dict(self.results)
        assert isinstance(d, dict)
        for key, val in d.items():
            assert isinstance(val, float)

    def test_to_dataframe(self):
        df = tm.to_dataframe(self.results)
        assert len(df) == 3
        assert "MSE" in df.index
        assert "MAE" in df.index
        assert "RMSE" in df.index

    def test_to_json_string(self):
        json_str = tm.to_json(self.results)
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "MSE" in data

    def test_to_json_file(self, tmp_path):
        path = str(tmp_path / "results.json")
        tm.to_json(self.results, path=path)
        with open(path) as f:
            data = json.load(f)
        assert "MSE" in data

    def test_to_csv_string(self):
        csv_str = tm.to_csv(self.results)
        assert isinstance(csv_str, str)
        assert "MSE" in csv_str

    def test_to_csv_file(self, tmp_path):
        path = str(tmp_path / "results.csv")
        tm.to_csv(self.results, path=path)
        import pandas as pd
        df = pd.read_csv(path, index_col=0)
        assert "MSE" in df.index

    def test_to_dict_with_dict_values(self):
        """Test export with dict-returning metrics (e.g., quantile_loss)."""
        samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, 50, self.C, self.T)
        ql = tm.prediction.quantile_loss(self.target, samples)
        results = {"QuantileLoss": ql, "MSE": torch.tensor(0.05)}
        d = tm.to_dict(results)
        assert isinstance(d["QuantileLoss"], dict)
        assert isinstance(d["MSE"], float)


class TestDieboldMariano:
    def setup_method(self):
        torch.manual_seed(42)
        self.T = 200
        self.target = torch.randn(self.T)
        self.forecast_a = self.target + 0.1 * torch.randn(self.T)  # good forecast
        self.forecast_b = self.target + 0.5 * torch.randn(self.T)  # bad forecast

    def test_dm_significant_difference(self):
        result = tm.diebold_mariano(self.target, self.forecast_a, self.forecast_b)
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert result["p_value"] < 0.05  # should detect difference
        assert result["significant"] == True

    def test_dm_no_difference(self):
        forecast_c = self.target + 0.1 * torch.randn(self.T)
        result = tm.diebold_mariano(self.target, self.forecast_a, forecast_c)
        assert result["p_value"] > 0.05  # should not detect difference

    def test_dm_mae_loss(self):
        result = tm.diebold_mariano(self.target, self.forecast_a, self.forecast_b, loss="mae")
        assert result["p_value"] < 0.05

    def test_dm_numpy_input(self):
        result = tm.diebold_mariano(
            self.target.numpy(), self.forecast_a.numpy(), self.forecast_b.numpy()
        )
        assert result["p_value"] < 0.05

    def test_paired_t_test(self):
        vals_a = [0.1, 0.12, 0.09, 0.11, 0.1]
        vals_b = [0.2, 0.22, 0.19, 0.21, 0.2]
        result = tm.paired_t_test(vals_a, vals_b)
        assert result["significant"] == True
        assert result["p_value"] < 0.05


class TestPerHorizon:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 2, 3, 12
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, 50, self.C, self.T)

    def test_per_horizon_mse(self):
        results = tm.per_horizon(tm.prediction.mse, self.target, self.forecast)
        assert len(results) == self.T
        for h in range(self.T):
            assert h in results
            assert results[h].shape == ()

    def test_per_horizon_subset(self):
        results = tm.per_horizon(
            tm.prediction.mse, self.target, self.forecast, horizons=[0, 5, 11]
        )
        assert len(results) == 3
        assert 0 in results
        assert 5 in results
        assert 11 in results

    def test_per_horizon_prob(self):
        results = tm.per_horizon_prob(
            tm.prediction.crps, self.target, self.samples
        )
        assert len(results) == self.T

    def test_horizon_summary(self):
        results = tm.per_horizon(tm.prediction.mse, self.target, self.forecast)
        summary = tm.horizon_summary(results)
        assert summary.shape == (self.T,)

    def test_per_horizon_with_mask(self):
        mask = torch.ones(self.B, self.C, self.T)
        mask[:, :, 6:] = 0
        results = tm.per_horizon(
            tm.prediction.mse, self.target, self.forecast, mask=mask
        )
        assert len(results) == self.T


class TestMultivariateMetrics:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.S = 50
        self.target = torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)

    def test_energy_score(self):
        val = tm.prediction.energy_score(self.target, self.samples)
        assert val.shape == ()
        assert val > 0

    def test_energy_score_perfect(self):
        """Energy score should be 0 when samples equal target."""
        perfect_samples = self.target.unsqueeze(1).expand(-1, self.S, -1, -1)
        val = tm.prediction.energy_score(self.target, perfect_samples)
        assert torch.allclose(val, torch.tensor(0.0), atol=1e-5)

    def test_variogram_score(self):
        val = tm.prediction.variogram_score(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_variogram_score_order(self):
        """Better samples should have lower variogram score."""
        good_samples = self.target.unsqueeze(1) + 0.05 * torch.randn(self.B, self.S, self.C, self.T)
        bad_samples = self.target.unsqueeze(1) + 1.0 * torch.randn(self.B, self.S, self.C, self.T)
        good_val = tm.prediction.variogram_score(self.target, good_samples)
        bad_val = tm.prediction.variogram_score(self.target, bad_samples)
        assert good_val < bad_val

    def test_energy_score_with_mask(self):
        mask = torch.ones(self.B, self.C, self.T)
        mask[:, :, 12:] = 0
        val = tm.prediction.energy_score(self.target, self.samples, mask=mask)
        assert val.shape == ()

    def test_calculator_multivariate(self):
        calc = tm.MetricCalculator(
            task="prediction", mode="probabilistic",
            metrics=["EnergyScore", "VariogramScore"]
        )
        results = calc.compute(self.target, self.samples)
        assert "EnergyScore" in results
        assert "VariogramScore" in results
