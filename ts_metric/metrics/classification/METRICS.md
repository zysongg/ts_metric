# Classification Metrics

时序分类指标，用于时序分类任务的评估。

## 输入约定

- **labels**: `(N,)` — 整数类别标签
- **preds**: `(N,)` — 预测类别标签，或 `(N, C)` 类别概率
- **average**: 多类平均方式，`"macro"`（默认）或 `"micro"`

---

## Accuracy

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

- **含义**: 正确预测的比例
- **范围**: $[0, 1]$
- **越大越好**

---

## Precision

### Macro Average（默认）

$$\text{Precision} = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FP_c}$$

其中 $TP_c = |\{i : \hat{y}_i = c \wedge y_i = c\}|$，$FP_c = |\{i : \hat{y}_i = c \wedge y_i \neq c\}|$。

### Micro Average

$$\text{Precision}_{micro} = \frac{\sum_c TP_c}{\sum_c (TP_c + FP_c)} = \text{Accuracy}$$

- **含义**: 预测为某类的样本中实际为该类的比例
- **范围**: $[0, 1]$
- **越大越好**
- **macro**: 每类 Precision 取均值，对少数类更公平
- **micro**: 等价于 Accuracy

---

## Recall

### Macro Average（默认）

$$\text{Recall} = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}$$

其中 $FN_c = |\{i : \hat{y}_i \neq c \wedge y_i = c\}|$。

### Micro Average

$$\text{Recall}_{micro} = \text{Accuracy}$$

- **含义**: 实际为某类的样本中被正确预测的比例
- **范围**: $[0, 1]$
- **越大越好**

---

## F1

### Macro Average（默认）

$$\text{F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

### Micro Average

$$\text{F1}_{micro} = \text{Accuracy}$$

- **含义**: Precision 和 Recall 的调和均值
- **范围**: $[0, 1]$
- **越大越好**

---

## AUC_ROC（二分类）

$$\text{AUC\_ROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

- **输入**: `labels` 为二值 (0/1)，`scores` 为 `(N,)` 或 `(N, 2)` 概率
- **含义**: ROC 曲线下面积
- **范围**: $[0, 1]$
- **越大越好**，0.5 表示随机猜测
- **注意**: 仅支持二分类

---

## 使用示例

```python
import ts_metric as tm

# 函数式 API
acc = tm.classification.accuracy(labels, preds)
f1 = tm.classification.f1(labels, preds, average="macro")

# Calculator API
calc = tm.MetricCalculator(task="classification", mode="default")
results = calc.compute_all(labels, preds)
# -> {"Accuracy": ..., "Precision": ..., "Recall": ..., "F1": ..., "AUC_ROC": ...}
```
