"""Tests for imputation metrics."""

import torch
import pytest
import ts_metric as tm


class TestImputationPoint:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 4, 3, 24
        self.target = torch.randn(self.B, self.C, self.T)
        self.forecast = self.target + 0.1 * torch.randn(self.B, self.C, self.T)
        # mask: 1 where ground truth is available for evaluation
        self.mask = torch.bernoulli(torch.ones(self.B, self.C, self.T) * 0.7)

    def test_mse(self):
        val = tm.imputation.mse(self.target, self.forecast, mask=self.mask)
        assert val.shape == ()
        assert val > 0

    def test_mre(self):
        val = tm.imputation.mre(self.target, self.forecast, mask=self.mask)
        assert val.shape == ()
        assert val > 0

    def test_mae(self):
        val = tm.imputation.mae(self.target, self.forecast, mask=self.mask)
        assert val.shape == ()
        assert val > 0

    def test_nd(self):
        val = tm.imputation.nd(self.target, self.forecast, mask=self.mask)
        assert val.shape == ()
        assert val > 0

    def test_perfect_imputation(self):
        val = tm.imputation.mse(self.target, self.target, mask=self.mask)
        assert torch.allclose(val, torch.tensor(0.0), atol=1e-7)


class TestImputationProbabilistic:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C, self.T = 2, 3, 24
        self.S = 50
        self.target = torch.randn(self.B, self.C, self.T)
        self.samples = self.target.unsqueeze(1) + 0.2 * torch.randn(self.B, self.S, self.C, self.T)
        self.mask = torch.bernoulli(torch.ones(self.B, self.C, self.T) * 0.7)

    def test_crps(self):
        val = tm.imputation.crps(self.target, self.samples, mask=self.mask)
        assert val.shape == ()
        assert val > 0

    def test_picp(self):
        val = tm.imputation.picp(self.target, self.samples, mask=self.mask)
        assert 0 <= val <= 1

    def test_interval_width(self):
        val = tm.imputation.interval_width(self.target, self.samples, mask=self.mask)
        assert val.shape == ()
        assert val > 0
