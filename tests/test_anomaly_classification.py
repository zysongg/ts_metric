"""Tests for anomaly detection metrics."""

import torch
import pytest
import ts_metric as tm


class TestAnomalyMetrics:
    def setup_method(self):
        torch.manual_seed(42)
        self.T = 100
        self.labels = torch.zeros(self.T)
        self.labels[20:30] = 1.0  # anomaly segment
        self.labels[60:65] = 1.0  # another anomaly segment

        self.preds_good = torch.zeros(self.T)
        self.preds_good[22] = 1.0  # detects first segment (one point)
        self.preds_good[60:65] = 1.0  # fully detects second segment

        self.preds_bad = torch.zeros(self.T)
        self.preds_bad[40:50] = 1.0  # false alarms

    def test_precision(self):
        val = tm.anomaly.precision(self.labels, self.preds_good)
        assert val.shape == ()
        assert val > 0

    def test_recall(self):
        val = tm.anomaly.recall(self.labels, self.preds_good)
        assert val.shape == ()

    def test_f1(self):
        val = tm.anomaly.f1(self.labels, self.preds_good)
        assert val.shape == ()

    def test_pa_f1_better_than_f1(self):
        """PA-F1 should be higher when partial detection covers whole segment."""
        f1_val = tm.anomaly.f1(self.labels, self.preds_good)
        pa_f1_val = tm.anomaly.pa_f1(self.labels, self.preds_good)
        assert pa_f1_val >= f1_val

    def test_pa_f1_perfect_detection(self):
        """PA-F1 should be 1.0 for perfect detection."""
        val = tm.anomaly.pa_f1(self.labels, self.labels)
        assert torch.allclose(val, torch.tensor(1.0), atol=1e-5)

    def test_pa_f1_no_detection(self):
        """PA-F1 should be 0 when no anomalies detected."""
        preds = torch.zeros(self.T)
        val = tm.anomaly.pa_f1(self.labels, preds)
        assert torch.allclose(val, torch.tensor(0.0), atol=1e-5)

    def test_auc_roc(self):
        scores = self.labels + 0.1 * torch.randn(self.T)
        val = tm.anomaly.auc_roc(self.labels, scores)
        assert val.shape == ()
        assert 0 <= val <= 1
        assert val > 0.8

    def test_auc_pr(self):
        scores = self.labels + 0.1 * torch.randn(self.T)
        val = tm.anomaly.auc_pr(self.labels, scores)
        assert val.shape == ()
        assert 0 <= val <= 1

    def test_2d_input(self):
        labels_2d = self.labels.unsqueeze(0).expand(3, -1)
        preds_2d = self.preds_good.unsqueeze(0).expand(3, -1)
        val = tm.anomaly.pa_f1(labels_2d, preds_2d)
        assert val.shape == ()

    def test_mask(self):
        mask = torch.zeros(self.T)
        mask[:50] = 1.0
        val = tm.anomaly.precision(self.labels, self.preds_good, mask=mask)
        assert val.shape == ()

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            tm.anomaly.precision(self.labels, self.labels[:50])

    def test_calculator(self):
        calc = tm.MetricCalculator(task="anomaly", mode="default", metrics=["PA_F1", "AUC_ROC"])
        scores = self.labels + 0.1 * torch.randn(self.T)
        results = calc.compute(self.labels, scores)
        assert "PA_F1" in results
        assert "AUC_ROC" in results


class TestClassificationMetrics:
    def setup_method(self):
        torch.manual_seed(42)
        self.N = 100
        self.labels = torch.randint(0, 3, (self.N,))
        self.preds = self.labels.clone()
        self.preds[:10] = (self.preds[:10] + 1) % 3  # 10 wrong predictions

    def test_accuracy(self):
        val = tm.classification.accuracy(self.labels, self.preds)
        assert val.shape == ()
        assert 0.8 < val < 1.0

    def test_precision(self):
        val = tm.classification.precision(self.labels, self.preds)
        assert val.shape == ()
        assert 0 < val <= 1

    def test_recall(self):
        val = tm.classification.recall(self.labels, self.preds)
        assert val.shape == ()
        assert 0 < val <= 1

    def test_f1(self):
        val = tm.classification.f1(self.labels, self.preds)
        assert val.shape == ()
        assert 0 < val <= 1

    def test_perfect_classification(self):
        val = tm.classification.accuracy(self.labels, self.labels)
        assert torch.allclose(val, torch.tensor(1.0), atol=1e-5)

    def test_binary_auc_roc(self):
        labels_bin = torch.randint(0, 2, (self.N,))
        scores = labels_bin.float() + 0.1 * torch.randn(self.N)
        val = tm.classification.auc_roc(labels_bin, scores)
        assert val.shape == ()
        assert 0 <= val <= 1
        assert val > 0.8

    def test_scores_2d(self):
        """Test with probability scores (N, C) instead of class predictions."""
        scores = torch.randn(self.N, 3)
        scores[range(self.N), self.labels] += 5.0  # boost correct class
        val = tm.classification.accuracy(self.labels, scores)
        assert val > 0.8

    def test_calculator(self):
        calc = tm.MetricCalculator(task="classification", mode="default", metrics=["Accuracy", "F1"])
        results = calc.compute(self.labels, self.preds)
        assert "Accuracy" in results
        assert "F1" in results
