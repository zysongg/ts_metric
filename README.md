# ts-metric

Time series metric computation library for **prediction**, **imputation**, **generation**, **anomaly detection**, and **classification** tasks.

PyTorch backend. Supports both **point** (regression) and **probabilistic** (distribution) evaluation modes.

## 安装

```bash
pip install git+https://github.com/zysongg/ts-metric.git
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
import ts_metric as tm

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
from ts_metric import MetricCalculator

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
from ts_metric import list_available_metrics
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
| probabilistic | CRPS, CRPS_sum, CRPS_exact, CRPS_sum_exact, MAE_Coverage, MSIS, PICP, QICE, MSE_median, MAE_median, LogLikelihood, EnergyScore, VariogramScore | 概率预测 |

### Imputation（插补）

| Mode | 指标 | 说明 |
|------|------|------|
| point | MSE, RMSE, MAE, MAPE, MRE, sMAPE, ND | 点插补 |
| probabilistic | CRPS, PICP, QICE, IntervalWidth | 概率插补 |

### Generation（生成，TSGBench 风格）

| 类别 | 指标 | 说明 |
|------|------|------|
| Feature-based | MDD, ACD, SD, KD | 边际分布、自相关、偏度、峰度差异 |
| Distance-based | ED, DTW | 配对样本欧氏距离、DTW 距离 |
| Model-based | DS, PS, C_FID | 判别分数、预测分数、Context-FID |

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

## 高级功能

### 指标评估可视化

```python
import ts_metric as tm

# Coverage plot: observed vs expected coverage
ax = tm.plot_coverage(target, samples)

# Quantile loss per level
ax = tm.plot_quantile_loss(target, samples)

# Calibration (reliability) diagram
ax = tm.plot_calibration(target, samples)

# Compare CRPS across multiple models
samples_dict = {"Model A": samples_a, "Model B": samples_b}
ax = tm.plot_crps_comparison(target, samples_dict)
```

可选依赖：`pip install matplotlib`

### 序列可视化

```python
import ts_metric as tm

# Point prediction with lookback
tm.plot_prediction(forecast, target, inputs, sample_id=0, channel=0)

# Multi-channel prediction
tm.plot_prediction_multi(forecast, target, inputs, channels=[0, 1, 2])

# Probabilistic prediction with confidence intervals
tm.plot_prob_prediction(samples, target, inputs, cl=(0.5, 0.9))

# Point imputation (no lookback)
tm.plot_imputation(imputed, target, mask, sample_id=0, channel=0)

# Multi-channel imputation
tm.plot_imputation_multi(imputed, target, mask, channels=[0, 1, 2])

# Probabilistic imputation with confidence intervals
tm.plot_prob_imputation(samples, target, mask, cl=(0.5, 0.9))
```

所有绘图使用 Times New Roman 字体，支持自定义 `ax` 参数以组合多图。

### 导出格式

```python
calc = tm.MetricCalculator(task="prediction", mode="point")
results = calc.compute(target, forecast)

# Dict
d = tm.to_dict(results)          # {"MSE": 0.01, "MAE": 0.08, ...}

# pandas DataFrame
df = tm.to_dataframe(results)    # metric | value

# JSON
tm.to_json(results, path="results.json")

# CSV
tm.to_csv(results, path="results.csv")
```

### Diebold-Mariano 检验

比较两个预测模型是否有**统计显著**差异：

```python
result = tm.diebold_mariano(target, forecast_a, forecast_b, loss="mse")
print(result)
# {"statistic": -3.42, "p_value": 0.0006, "significant": True}

# 配对 t 检验（多次评估）
result = tm.paired_t_test(metric_values_a, metric_values_b)
```

### Per-Horizon 指标

按预测步长分解指标，分析不同 horizon 的表现：

```python
# Point metrics per horizon
results = tm.per_horizon(tm.prediction.mse, target, forecast)
# {0: tensor(0.01), 1: tensor(0.02), ..., T-1: tensor(0.05)}

# Probabilistic metrics per horizon
results = tm.per_horizon_prob(tm.prediction.crps, target, samples)

# Convert to tensor
summary = tm.horizon_summary(results)  # shape (T,)
```

### 多变量概率指标

```python
# Energy Score: multivariate CRPS
es = tm.prediction.energy_score(target, samples)

# Variogram Score: dependence structure quality
vs = tm.prediction.variogram_score(target, samples, p=2)
```

## 发布到 PyPI

### 1. 安装构建工具

```bash
pip install build twine
```

### 2. 构建分发包

```bash
python -m build
```

这会在 `dist/` 目录下生成 `.whl` 和 `.tar.gz` 文件。

### 3. 上传到 PyPI

```bash
# 上传到正式 PyPI
twine upload dist/*

# 或先测试上传到 TestPyPI
twine upload --repository testpypi dist/*
```

上传时需要输入 PyPI 用户名和密码（或 API token）。

### 4. 使用 API Token（推荐）

在 https://pypi.org/manage/account/token/ 创建 token，然后：

```bash
twine upload --username __token__ --password pypi-xxxxxxxx dist/*
```

### 5. 用户安装

```bash
# 从 PyPI 安装
pip install ts-metric

# 从 TestPyPI 安装
pip install --index-url https://test.pypi.org/simple/ ts-metric
```

## 运行测试

```bash
cd ts-metric
pytest tests/ -v
```
