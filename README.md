# timescore

Time series metric computation library for **prediction**, **imputation**, **generation**, **anomaly detection**, and **classification** tasks.

PyTorch backend. Supports both **point** (regression) and **probabilistic** (distribution) evaluation modes.

## 安装

```bash
pip install git+https://github.com/zysongg/timescore.git
```

## 输入形状约定

| 任务 | 输入 | Shape |
|------|------|-------|
| Prediction / Imputation (point) | `target`, `forecast` | `(B, C, T)` 或 `(C, T)` |
| Prediction / Imputation (probabilistic) | `target` | `(B, C, T)` 或 `(C, T)` |
| Prediction / Imputation (probabilistic) | `samples` | `(B, S, C, T)` 或 `(S, C, T)` |
| Generation | `real` | `(N, C, T)` 或 `(C, T)` |
| Generation | `generated` | `(M, C, T)` 或 `(C, T)` |
| Anomaly Detection | `labels`, `preds` / `scores` | `(B, T)` 或 `(T,)` |
| Classification | `labels`, `preds` | `(N,)` |
| Classification | `scores` | `(N, C)` |

- **B**: batch size
- **C**: number of features (channels)
- **T**: time steps
- **S**: number of probabilistic samples
- **N/M**: number of samples

## Mask 支持

Prediction / Imputation / Generation 指标支持可选 `mask` 参数：

- `mask=1` 表示有效时间步（参与计算）
- `mask=0` 表示屏蔽（不参与计算）
- 支持广播：`(T,)`, `(C, T)`, `(B, T)`, `(B, 1, T)`, `(B, C, T)`

## 快速开始

### Prediction

```python
import torch
import timescore as tm

B, C, T = 4, 3, 24
target = torch.randn(B, C, T)
forecast = target + 0.1 * torch.randn(B, C, T)

# Point metrics
mse = tm.prediction.mse(target, forecast)
nrmse = tm.prediction.nrmse(target, forecast)

# Probabilistic metrics
samples = target.unsqueeze(1) + 0.2 * torch.randn(B, 50, C, T)
crps = tm.prediction.crps(target, samples)         # K2VAE/DeepAR convention
crps_exact = tm.prediction.crps_exact(target, samples)  # exact formula

# With mask
mask = torch.ones(B, C, T)
mask[:, :, 12:] = 0
mse_masked = tm.prediction.mse(target, forecast, mask=mask)
```

### Imputation

```python
# Reuses prediction metrics + MRE
mse = tm.imputation.mse(target, imputed, mask=missing_mask)
mre = tm.imputation.mre(target, imputed, mask=missing_mask)

# Probabilistic
crps = tm.imputation.crps(target, samples, mask=missing_mask)
width = tm.imputation.interval_width(target, samples, mask=missing_mask)
```

### Generation

```python
real = torch.randn(100, C, T)
generated = torch.randn(80, C, T)

# Point metrics
fid = tm.generation.fidelity(real, generated)
ds = tm.generation.discriminative_score(real, generated)

# Probabilistic metrics
mmd = tm.generation.mmd(real, generated)
js = tm.generation.js_divergence(real, generated)
```

### Anomaly Detection

```python
labels = torch.zeros(100)
labels[20:30] = 1.0  # anomaly segment
preds = torch.zeros(100)
preds[22] = 1.0       # detects the segment

# Standard metrics
f1 = tm.anomaly.f1(labels, preds)

# Point-Adjust F1 (standard for time series anomaly detection)
pa_f1 = tm.anomaly.pa_f1(labels, preds)

# AUC metrics (requires anomaly scores)
scores = labels + 0.1 * torch.randn(100)
auc = tm.anomaly.auc_roc(labels, scores)
```

### Classification

```python
labels = torch.randint(0, 3, (100,))
preds = labels.clone()
preds[:10] = (preds[:10] + 1) % 3

acc = tm.classification.accuracy(labels, preds)
f1 = tm.classification.f1(labels, preds, average="macro")
```

### MetricCalculator API

```python
from timescore import MetricCalculator

# Prediction point metrics
calc = MetricCalculator(task="prediction", mode="point", metrics=["MSE", "NRMSE"])
results = calc.compute(target, forecast)

# Anomaly detection
calc = MetricCalculator(task="anomaly", mode="default")
results = calc.compute_all(labels, scores)

# Classification
calc = MetricCalculator(task="classification", mode="default", metrics=["Accuracy", "F1"])
results = calc.compute(labels, preds)

# Per-feature breakdown (prediction/imputation only)
per_feat = calc.compute_per_feature(target, forecast)

# List all available metrics
from timescore import list_available_metrics
print(list_available_metrics())
```

## CRPS 命名说明

| 函数 | 公式 | 用途 |
|---|---|---|
| `crps()` | mean(wQuantileLoss) | **默认**，与 K2VAE/DeepAR 论文一致 |
| `crps_sum()` | mean(wQL) on Σ_c | 与 K2VAE CRPS-Sum 一致 |
| `crps_exact()` | E\|X-y\| - ½E\|X-X'\| | 精确 CRPS，数学上更严格 |
| `crps_sum_exact()` | exact CRPS on Σ_c | 精确版 CRPS-Sum |

投稿和论文对比时，使用 `crps()` 和 `crps_sum()` 与已发表论文对齐。

## 可用指标一览

### Prediction（预测）

| Mode | 指标 | 说明 |
|------|------|------|
| point | MSE, RMSE, NRMSE, MAE, MAPE, sMAPE, ND, R2, Correlation | 点预测 |
| probabilistic | CRPS, CRPS_sum, CRPS_exact, CRPS_sum_exact, MAE_Coverage, MSIS, PICP, QICE, MSE_median, MAE_median, LogLikelihood | 概率预测 |

### Imputation（插补）

| Mode | 指标 | 说明 |
|------|------|------|
| point | MSE, RMSE, MAE, MAPE, MRE, sMAPE, ND | 点插补 |
| probabilistic | CRPS, PICP, QICE, IntervalWidth | 概率插补 |

### Generation（生成）

| Mode | 指标 | 说明 |
|------|------|------|
| point | Fidelity, DiscriminativeScore, Correlation, KLDivergence | 样本级 |
| probabilistic | MMD, JSDivergence, LogLikelihood | 分布级 |

### Anomaly Detection（异常检测）

| Mode | 指标 | 说明 |
|------|------|------|
| default | Precision, Recall, F1, PA_Precision, PA_Recall, **PA_F1**, AUC_ROC, AUC_PR | 含 Point-Adjust F1 |

### Classification（分类）

| Mode | 指标 | 说明 |
|------|------|------|
| default | Accuracy, Precision, Recall, F1, AUC_ROC | 支持 macro/micro |

## GluonTS / K2VAE 兼容性

以下指标已通过与 GluonTS Evaluator 和 K2VAE Evaluator 的数值对比验证：

- `QuantileLoss`, `wQuantileLoss`, `Coverage` — 与 GluonTS 完全一致
- `MSE`, `MAPE`, `sMAPE`, `MSIS`, `NRMSE` — 与 GluonTS 完全一致
- `CRPS`, `CRPS_sum` — 与 K2VAE Evaluator 一致（<2% 聚合差异）

## 运行测试

```bash
cd timescore
pytest tests/ -v
```
