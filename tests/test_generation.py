"""Tests for generation metrics."""

import torch
import pytest
import timescore as tm


class TestGenerationPoint:
    def setup_method(self):
        torch.manual_seed(42)
        self.C, self.T = 3, 24
        self.real = torch.randn(100, self.C, self.T)
        self.generated = torch.randn(80, self.C, self.T)  # M may != N

    def test_fidelity(self):
        val = tm.generation.fidelity(self.real, self.generated)
        assert val.shape == ()
        assert val > 0

    def test_correlation(self):
        val = tm.generation.correlation(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_kl_divergence(self):
        val = tm.generation.kl_divergence(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_discriminative_score(self):
        val = tm.generation.discriminative_score(self.real, self.generated)
        assert val.shape == ()
        assert 0 <= val <= 0.5

    def test_same_distribution(self):
        """Generated from same distribution should have low fidelity distance."""
        real2 = torch.randn(100, self.C, self.T)
        val = tm.generation.fidelity(self.real, real2)
        assert val < tm.generation.fidelity(self.real, self.generated + 5.0)


class TestGenerationProbabilistic:
    def setup_method(self):
        torch.manual_seed(42)
        self.C, self.T = 3, 24
        self.real = torch.randn(100, self.C, self.T)
        self.generated = torch.randn(80, self.C, self.T)

    def test_mmd(self):
        val = tm.generation.mmd(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_js_divergence(self):
        val = tm.generation.js_divergence(self.real, self.generated)
        assert val.shape == ()
        assert val >= 0

    def test_log_likelihood(self):
        val = tm.generation.log_likelihood(self.real, self.generated)
        assert val.shape == ()

    def test_identical_samples(self):
        """MMD should be ~0 when real == generated."""
        val = tm.generation.mmd(self.real, self.real)
        assert val < 0.01
