# Anomaly Detection Metrics

时序异常检测指标，支持二值标签和异常分数两种输入。

## 输入约定

- **labels**: `(B, T)` 或 `(T,)` — 二值标签，1=异常，0=正常
- **preds**: `(B, T)` 或 `(T,)` — 二值预测
- **scores**: `(B, T)` 或 `(T,)` — 异常分数（越高越异常）
- **mask**: 可选，与 labels 同 shape 或可广播，1=有效，0=屏蔽

---

## 标准二值指标

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **含义**: 预测为异常的样本中，实际为异常的比例
- **范围**: $[0, 1]$
- **越大越好**

### Recall

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **含义**: 实际异常的样本中，被正确检出的比例
- **范围**: $[0, 1]$
- **越大越好**

### F1

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **含义**: Precision 和 Recall 的调和均值
- **范围**: $[0, 1]$
- **越大越好**

---

## 时序专属指标：Point-Adjust

### Point-Adjust 机制

时序异常通常以连续段出现。Point-Adjust (PA) 规则：

> 如果一个连续异常段中**至少有一个点**被检出，则该段**所有点**都视为正确检出。

这是时序异常检测领域的标准做法，因为运维人员通常只需要一个告警就能发现整段异常。

### PA_Precision

$$\text{PA\_Precision} = \frac{TP_{pa}}{TP_{pa} + FP_{pa}}$$

其中 $TP_{pa}$ 和 $FP_{pa}$ 是 Point-Adjust 后的 True Positive 和 False Positive。

- **含义**: PA 调整后的精确率
- **范围**: $[0, 1]$
- **越大越好**

### PA_Recall

$$\text{PA\_Recall} = \frac{TP_{pa}}{TP_{pa} + FN_{pa}}$$

- **含义**: PA 调整后的召回率
- **范围**: $[0, 1]$
- **越大越好**
- **特点**: 由于 PA 机制，PA_Recall 通常高于标准 Recall

### PA_F1

$$\text{PA\_F1} = \frac{2 \cdot \text{PA\_Precision} \cdot \text{PA\_Recall}}{\text{PA\_Precision} + \text{PA\_Recall}}$$

- **含义**: 时序异常检测的标准指标
- **范围**: $[0, 1]$
- **越大越好**
- **参考**: 几乎所有时序异常检测论文都报告此指标

---

## AUC 指标

### AUC_ROC

$$\text{AUC\_ROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

其中 $\text{TPR} = \text{Recall}$，$\text{FPR} = \frac{FP}{FP + TN}$。

通过梯形法则对排序后的阈值进行数值积分。

- **含义**: ROC 曲线下面积，衡量不同阈值下的整体分类性能
- **范围**: $[0, 1]$
- **越大越好**，0.5 表示随机猜测

### AUC_PR

$$\text{AUC\_PR} = \int_0^1 \text{Precision}(\text{Recall}) \, d(\text{Recall})$$

- **含义**: Precision-Recall 曲线下面积
- **范围**: $[0, 1]$
- **越大越好**
- **优势**: 对类别不平衡（正常 >> 异常）更敏感

---

## Mask 支持

所有指标支持可选 `mask` 参数：
- `mask=1` 的位置参与计算
- `mask=0` 的位置被忽略
- 常用于排除 warm-up 期或标签不可靠的时间步
