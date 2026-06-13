# Imputation Metrics

## Point Metrics (回归指标)

复用 prediction 模块的指标：**MSE, MAE, RMSE, MAPE, sMAPE, ND**

详见 `../prediction/METRICS.md`

### MRE (Mean Relative Error)

$$\text{MRE} = \frac{\text{mean}(|y - \hat{y}|)}{\text{mean}(|y|)} = \frac{\frac{1}{N} \sum_{b,c,t} |y_{b,c,t} - \hat{y}_{b,c,t}|}{\frac{1}{N} \sum_{b,c,t} |y_{b,c,t}|}$$

- **含义**: 平均相对误差，MAE 与真实值均值的比值
- **范围**: $[0, +\infty)$
- **越小越好**
- **与 ND 的区别**: MRE 是均值之比，ND 是总和之比。当 mask 不均匀时结果不同

### 插补任务的特殊性

在插补任务中：
- `target`: 真实完整序列（ground truth）
- `forecast`: 插补后的序列
- `mask`: 标记**有真实值可评估**的位置（`mask=1` 表示该位置有 ground truth）

典型用法：
```python
# mask=1 表示该位置被遮盖（缺失），有 ground truth 可评估
mse = tm.imputation.mse(target, imputed, mask=missing_mask)
```

---

## Probabilistic Metrics (概率指标)

复用 prediction 模块的指标：**CRPS, PICP, QICE**

详见 `../prediction/METRICS.md`

### Interval Width

$$\text{IntervalWidth} = \frac{1}{B \cdot C \cdot T} \sum_{b,c,t} \left( \hat{y}^{(1-\alpha/2)}_{b,c,t} - \hat{y}^{(\alpha/2)}_{b,c,t} \right)$$

其中 $\hat{y}^{(q)}$ 是样本的 $q$ 分位数，$\alpha$ 是显著性水平（默认 0.1 表示 90% 区间）。

- **含义**: 预测区间的平均宽度
- **越小越好**（在保持覆盖率的前提下）
- **与 PICP 配合使用**: PICP 高 + IntervalWidth 小 = 好的不确定性估计

---

## Mask 支持

所有指标支持可选 `mask` 参数（shape: `(B, C, T)` 或可广播）：
- `mask=1` 的位置参与计算（有 ground truth 的位置）
- `mask=0` 的位置被忽略（无需评估的位置）
- 自动归一化：$\text{MSE} = \frac{\sum m \cdot (y - \hat{y})^2}{\sum m}$
