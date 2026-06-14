"""Tests for generation metrics (TSGBench-style)."""

import torch
import pytest
import ts_metric as tm


class TestFeatureBased:
    def setup_method(self):
        torch.manual_seed(42)
        self.C, self.T = 3, 24
        self.real = torch.randn(100, self.C, self.T)
        self.generated = torch.randn(80, self.C, self.T)

    def test_mdd(self):
        val = tm.generation.mdd(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_mdd_same_distribution(self):
        val = tm.generation.mdd(self.real, self.real)
        assert val < 0.01

    def test_acd(self):
        val = tm.generation.acd(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_sd(self):
        val = tm.generation.sd(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_kd(self):
        val = tm.generation.kd(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_feature_metrics_worse_for_shifted(self):
        """Metrics should be larger when generated is shifted."""
        shifted = self.real + 5.0
        assert tm.generation.sd(self.real, shifted) > tm.generation.sd(self.real, self.real)
        assert tm.generation.kd(self.real, shifted) > tm.generation.kd(self.real, self.real)


class TestDistanceBased:
    def setup_method(self):
        torch.manual_seed(42)
        self.C, self.T = 3, 24
        self.real = torch.randn(50, self.C, self.T)
        self.generated = torch.randn(50, self.C, self.T)

    def test_ed(self):
        val = tm.generation.ed(self.real, self.generated)
        assert val.shape == ()
        assert val > 0

    def test_ed_identical(self):
        val = tm.generation.ed(self.real, self.real)
        assert torch.allclose(val, torch.tensor(0.0), atol=1e-6)

    def test_ed_different_sizes(self):
        gen_small = torch.randn(30, self.C, self.T)
        val = tm.generation.ed(self.real, gen_small)
        assert val.shape == ()
        assert val > 0


class TestModelBased:
    def setup_method(self):
        torch.manual_seed(42)
        self.C, self.T = 3, 24
        self.real = torch.randn(50, self.C, self.T)
        self.generated = self.real + 0.1 * torch.randn(50, self.C, self.T)

    def test_ds(self):
        val = tm.generation.ds(self.real, self.generated, iterations=100)
        assert val.shape == ()
        assert 0 <= val <= 0.5

    def test_ps(self):
        val = tm.generation.ps(self.real, self.generated, iterations=100)
        assert val.shape == ()
        assert val >= 0

    def test_c_fid_fallback(self):
        """Test C-FID with fallback encoder (no ts2vec)."""
        val = tm.generation.c_fid(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_c_fid_identical(self):
        val = tm.generation.c_fid(self.real, self.real)
        assert val < 1.0


class TestCalculator:
    def setup_method(self):
        torch.manual_seed(42)
        self.C, self.T = 3, 24
        self.real = torch.randn(50, self.C, self.T)
        self.generated = torch.randn(40, self.C, self.T)

    def test_calculator_feature_metrics(self):
        calc = tm.MetricCalculator(
            task="generation", mode="default",
            metrics=["MDD", "ACD", "SD", "KD", "ED"]
        )
        results = calc.compute(self.real, self.generated)
        assert "MDD" in results
        assert "ACD" in results
        assert "SD" in results
        assert "KD" in results
        assert "ED" in results

    def test_shape_mismatch(self):
        bad_gen = torch.randn(40, 5, self.T)  # different C
        with pytest.raises(ValueError):
            tm.generation.mdd(self.real, bad_gen)
