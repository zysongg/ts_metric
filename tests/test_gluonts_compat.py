"""Tests verifying timescore against GluonTS for consistency."""

import torch
import numpy as np
import pytest
import ts_metric as tm


class TestGluonTSCompatibility:
    """Verify our metrics match GluonTS implementations."""

    def setup_method(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.S = 50
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)

    def test_quantile_loss_gluonts(self):
        """Verify quantile_loss matches GluonTS formula: 2*sum(|(ŷ-y)*(𝟙[y≤ŷ]-q)|)."""
        from gluonts.evaluation.metrics import quantile_loss as gluonts_ql

        q = 0.5
        qhat = torch.quantile(self.samples, q, dim=1).numpy()
        target_np = self.target.numpy()

        our_ql = tm.prediction.quantile_loss(self.target, self.samples, quantile_levels=[q])
        gluonts_ql_val = gluonts_ql(target_np.flatten(), qhat.flatten(), q)

        assert abs(our_ql[q].item() - gluonts_ql_val) < 1e-3, \
            f"QuantileLoss mismatch: ours={our_ql[q].item():.6f}, gluonts={gluonts_ql_val:.6f}"

    def test_coverage_gluonts(self):
        """Verify coverage matches GluonTS: mean(y ≤ ŷ_q)."""
        from gluonts.evaluation.metrics import coverage as gluonts_cov

        q = 0.5
        qhat = torch.quantile(self.samples, q, dim=1).numpy()
        target_np = self.target.numpy()

        our_cov = tm.prediction.coverage(self.target, self.samples, quantile_levels=[q])
        gluonts_cov_val = gluonts_cov(target_np.flatten(), qhat.flatten())

        assert abs(our_cov[q].item() - gluonts_cov_val) < 1e-3, \
            f"Coverage mismatch: ours={our_cov[q].item():.6f}, gluonts={gluonts_cov_val:.6f}"

    def test_mse_gluonts(self):
        """Verify MSE matches GluonTS."""
        from gluonts.evaluation.metrics import mse as gluonts_mse

        target_np = self.target.numpy()
        forecast_np = self.forecast.numpy()

        our_mse = tm.prediction.mse(self.target, self.forecast).item()
        gluonts_mse_val = gluonts_mse(target_np.flatten(), forecast_np.flatten())

        assert abs(our_mse - gluonts_mse_val) < 1e-5, \
            f"MSE mismatch: ours={our_mse:.6f}, gluonts={gluonts_mse_val:.6f}"

    def test_mape_gluonts(self):
        """Verify MAPE matches GluonTS."""
        from gluonts.evaluation.metrics import mape as gluonts_mape

        target_np = self.target.numpy()
        forecast_np = self.forecast.numpy()

        our_mape = tm.prediction.mape(self.target, self.forecast).item()
        gluonts_mape_val = gluonts_mape(target_np.flatten(), forecast_np.flatten())

        assert abs(our_mape - gluonts_mape_val) < 1e-3, \
            f"MAPE mismatch: ours={our_mape:.6f}, gluonts={gluonts_mape_val:.6f}"

    def test_smape_gluonts(self):
        """Verify sMAPE matches GluonTS."""
        from gluonts.evaluation.metrics import smape as gluonts_smape

        target_np = self.target.numpy()
        forecast_np = self.forecast.numpy()

        our_smape = tm.prediction.smape(self.target, self.forecast).item()
        gluonts_smape_val = gluonts_smape(target_np.flatten(), forecast_np.flatten())

        assert abs(our_smape - gluonts_smape_val) < 1e-3, \
            f"sMAPE mismatch: ours={our_smape:.6f}, gluonts={gluonts_smape_val:.6f}"

    def test_msis_gluonts(self):
        """Verify MSIS matches GluonTS formula."""
        from gluonts.evaluation.metrics import msis as gluonts_msis

        alpha = 0.05
        lower = torch.quantile(self.samples, alpha / 2, dim=1).numpy()
        upper = torch.quantile(self.samples, 1 - alpha / 2, dim=1).numpy()
        target_np = self.target.numpy()

        seasonal_error = np.mean(np.abs(target_np[:, :, 1:] - target_np[:, :, :-1]))

        our_msis = tm.prediction.msis(
            self.target, self.samples, alpha=alpha, seasonal_error=seasonal_error
        ).item()
        gluonts_msis_val = gluonts_msis(
            target_np.flatten(), lower.flatten(), upper.flatten(),
            seasonal_error, alpha
        )

        assert abs(our_msis - gluonts_msis_val) / max(abs(gluonts_msis_val), 1e-6) < 0.01, \
            f"MSIS mismatch: ours={our_msis:.6f}, gluonts={gluonts_msis_val:.6f}"


class TestNewMetrics:
    """Test newly added GluonTS-style metrics."""

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.S = 50
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)

    def test_nrmse(self):
        val = tm.prediction.nrmse(self.target, self.forecast)
        assert val.shape == ()
        assert val > 0
        rmse_val = tm.prediction.rmse(self.target, self.forecast)
        expected = rmse_val / torch.abs(self.target).mean()
        assert torch.allclose(val, expected, atol=1e-5)

    def test_quantile_loss_all_levels(self):
        result = tm.prediction.quantile_loss(self.target, self.samples)
        assert len(result) == 9
        for q, val in result.items():
            assert val.shape == ()
            assert val >= 0

    def test_w_quantile_loss(self):
        result = tm.prediction.w_quantile_loss(self.target, self.samples)
        assert len(result) == 9
        for q, val in result.items():
            assert val.shape == ()
            assert val >= 0

    def test_crps(self):
        val = tm.prediction.crps(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_coverage_all_levels(self):
        result = tm.prediction.coverage(self.target, self.samples)
        assert len(result) == 9
        for q, val in result.items():
            assert 0 <= val <= 1

    def test_mae_coverage(self):
        val = tm.prediction.mae_coverage(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_msis(self):
        val = tm.prediction.msis(self.target, self.samples)
        assert val.shape == ()
        assert val >= 0

    def test_msis_custom_seasonal_error(self):
        val = tm.prediction.msis(self.target, self.samples, seasonal_error=1.0)
        assert val.shape == ()

    def test_calculator_new_metrics(self):
        calc = tm.MetricCalculator(
            task="prediction", mode="probabilistic",
            metrics=["CRPS", "MAE_Coverage", "MSIS"]
        )
        results = calc.compute(self.target, self.samples)
        assert "CRPS" in results
        assert "MAE_Coverage" in results
        assert "MSIS" in results


class TestK2VAECompatibility:
    """Verify our CRPS_k2vae and CRPS_sum_k2vae match K2VAE's evaluator."""

    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 4, 3, 24
        self.S = 100
        self.target = torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)

    def test_crps_equals_mean_wql(self):
        """crps should be identical to internal _mean_w_quantile_loss."""
        from timescore.metrics.prediction.probabilistic import _mean_w_quantile_loss
        val_crps = tm.prediction.crps(self.target, self.samples)
        val_wql = _mean_w_quantile_loss(self.target, self.samples)
        assert torch.allclose(val_crps, val_wql, atol=1e-7)

    def test_crps_vs_k2vae_evaluator(self):
        """Verify crps matches K2VAE's Evaluator CRPS computation."""
        import sys
        sys.path.insert(0, "/data/songzy/workshop/project_ts/K2VAE")
        from probts.utils.evaluator import Evaluator as K2Evaluator

        # K2VAE expects: targets (B, T, C), forecasts (B, S, T, C)
        targets_k2 = self.target.permute(0, 2, 1).numpy()
        forecasts_k2 = self.samples.permute(0, 1, 3, 2).numpy()

        evaluator = K2Evaluator(quantiles_num=10)
        # get_sequence_metrics per sample, then mean
        k2_crps_vals = []
        for i in range(self.B):
            m = evaluator.get_sequence_metrics(
                targets_k2[i:i+1], forecasts_k2[i:i+1]
            )
            k2_crps_vals.append(m["CRPS"])
        k2_crps = np.mean(k2_crps_vals)

        our_crps = tm.prediction.crps(self.target, self.samples).item()

        assert abs(our_crps - k2_crps) / max(abs(k2_crps), 1e-8) < 0.01, \
            f"CRPS mismatch: ours={our_crps:.6f}, K2VAE={k2_crps:.6f}"

    def test_crps_sum_vs_k2vae_evaluator(self):
        """Verify crps_sum approximately matches K2VAE's CRPS-Sum.

        Note: K2VAE computes per-sequence then averages: mean(QL_i / |y_i|).
        We compute globally: sum(QL) / sum(|y|). These differ slightly (~1%).
        """
        import sys
        sys.path.insert(0, "/data/songzy/workshop/project_ts/K2VAE")
        from probts.utils.evaluator import Evaluator as K2Evaluator

        targets_k2 = self.target.permute(0, 2, 1).numpy()
        forecasts_k2 = self.samples.permute(0, 1, 3, 2).numpy()

        evaluator = K2Evaluator(quantiles_num=10)
        targets_sum = targets_k2.sum(axis=-1, keepdims=True)
        forecasts_sum = forecasts_k2.sum(axis=-1, keepdims=True)

        k2_crps_sum_vals = []
        for i in range(self.B):
            m = evaluator.get_sequence_metrics(
                targets_sum[i:i+1], forecasts_sum[i:i+1]
            )
            k2_crps_sum_vals.append(m["CRPS"])
        k2_crps_sum = np.mean(k2_crps_sum_vals)

        our_crps_sum = tm.prediction.crps_sum(self.target, self.samples).item()

        # ~2% tolerance due to per-sequence vs global aggregation difference
        assert abs(our_crps_sum - k2_crps_sum) / max(abs(k2_crps_sum), 1e-8) < 0.02, \
            f"CRPS_sum mismatch: ours={our_crps_sum:.6f}, K2VAE={k2_crps_sum:.6f}"
