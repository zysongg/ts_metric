"""Tests for prediction metrics."""

import torch
import pytest
import ts_metric as tm


class TestPredictionPoint:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 4, 3, 24
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)

    def test_mse(self):
        val = tm.prediction.mse(self.target, self.forecast)
        assert val.shape == ()
        assert val > 0

    def test_mae(self):
        val = tm.prediction.mae(self.target, self.forecast)
        assert val.shape == ()
        assert val > 0

    def test_rmse(self):
        val = tm.prediction.rmse(self.target, self.forecast)
        assert val.shape == ()
        assert torch.allclose(val, torch.sqrt(tm.prediction.mse(self.target, self.forecast)))

    def test_smape(self):
        val = tm.prediction.smape(self.target, self.forecast)
        assert val.shape == ()
        assert 0 <= val <= 2

    def test_r2(self):
        val = tm.prediction.r2(self.target, self.forecast)
        assert val.shape == ()
        assert val > 0.9  # small noise -> high R2

    def test_perfect_forecast(self):
        val = tm.prediction.mse(self.target, self.target)
        assert torch.allclose(val, torch.tensor(0.0), atol=1e-7)

    def test_2d_input(self):
        target_2d = self.target[0]  # (C, T)
        forecast_2d = self.forecast[0]
        val = tm.prediction.mse(target_2d, forecast_2d)
        val_3d = tm.prediction.mse(self.target[:1], self.forecast[:1])
        assert torch.allclose(val, val_3d, atol=1e-7)

    def test_mask(self):
        mask = torch.zeros(self.B, self.C, self.T)
        mask[:, :, :12] = 1.0  # only first half valid
        val_masked = tm.prediction.mse(self.target, self.forecast, mask=mask)
        val_first_half = tm.prediction.mse(self.target[:, :, :12], self.forecast[:, :, :12])
        assert torch.allclose(val_masked, val_first_half, atol=1e-6)

    def test_mask_broadcasting(self):
        mask_bt = torch.ones(self.B, self.T)
        val = tm.prediction.mse(self.target, self.forecast, mask=mask_bt)
        assert val.shape == ()

    def test_per_feature(self):
        vals = tm.prediction.mse_per_feature(self.target, self.forecast)
        assert vals.shape == (self.C,)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            tm.prediction.mse(self.target, self.forecast[:, :, :12])


class TestPredictionProbabilistic:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.S = 50
        self.target = torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)

    def test_crps(self):
        val = tm.prediction.crps(self.target, self.samples)
        assert val.shape == ()
        assert val > 0

    def test_picp(self):
        val = tm.prediction.picp(self.target, self.samples, alpha=0.1)
        assert 0 <= val <= 1

    def test_qice(self):
        val = tm.prediction.qice(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_mse_median(self):
        val = tm.prediction.mse_median(self.target, self.samples)
        assert val.shape == ()
        assert val > 0

    def test_log_likelihood(self):
        val = tm.prediction.log_likelihood(self.target, self.samples)
        assert val.shape == ()

    def test_3d_samples_input(self):
        samples_3d = self.samples[0]  # (S, C, T)
        target_2d = self.target[0]  # (C, T)
        val = tm.prediction.crps(target_2d, samples_3d)
        assert val.shape == ()
