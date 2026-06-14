# Generation Metrics (TSGBench-style)

时序生成评估指标，比较真实样本集合与生成样本集合。

## 输入约定

- **real**: `(N, C, T)` — N 条真实样本
- **generated**: `(M, C, T)` — M 条生成样本（M 可以不等于 N）

---

## Feature-based Metrics (统计特征指标)

### MDD (Marginal Distribution Distance)

$$\text{MDD} = \frac{1}{C \cdot T} \sum_{c,t} \int |p_{real}(x; c, t) - p_{gen}(x; c, t)| \, dx$$

通过直方图近似计算每个特征-时间步的边际分布距离。

- **含义**: 边际分布差异，衡量生成分布是否覆盖了真实分布的每个维度
- **越小越好**

### ACD (Auto-Correlation Distance)

$$\text{ACD} = \frac{1}{C} \sum_c \sqrt{\sum_{\text{lag}} (\text{ACF}_{real}(c, \text{lag}) - \text{ACF}_{gen}(c, \text{lag}))^2}$$

- **含义**: 自相关函数差异，衡量时序依赖结构的保持程度
- **越小越好**
- **max_lag**: 默认 min(64, T)

### SD (Skewness Difference)

$$\text{SD} = \frac{1}{C} \sum_c |\text{skew}_{real}(c) - \text{skew}_{gen}(c)|$$

- **含义**: 偏度差异，衡量分布对称性的保持
- **越小越好**

### KD (Kurtosis Difference)

$$\text{KD} = \frac{1}{C} \sum_c |\text{kurt}_{real}(c) - \text{kurt}_{gen}(c)|$$

- **含义**: 超额峰度差异，衡量分布尖锐度的保持
- **越小越好**

---

## Distance-based Metrics (距离指标)

### ED (Euclidean Distance)

$$\text{ED} = \frac{1}{P} \sum_{i=1}^{P} \frac{1}{C} \sum_c \|x^{real}_{i,c} - x^{gen}_{i,c}\|_2$$

- **含义**: 配对样本距离
- **越小越好**

### DTW (Dynamic Time Warping)

- **含义**: 配对样本 DTW 距离，允许时间轴对齐
- **越小越好**
- **可选依赖**: `pip install dtaidistance`

---

## Model-based Metrics (模型评估指标)

### DS (Discriminative Score)

$$\text{DS} = |\text{Accuracy} - 0.5|$$

- **含义**: GRU 分类器区分真实与生成的能力
- **范围**: $[0, 0.5]$，**越小越好**

### PS (Predictive Score)

$$\text{PS} = \text{MAE}_{real}$$

- **含义**: 在生成数据上训练的预测器在真实数据上的误差
- **越小越好**

### C-FID (Context-FID)

$$\text{C\_FID} = \|\mu_r - \mu_g\|^2 + \text{tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

- **含义**: 表示空间中的 Fréchet 距离
- **越小越好**
- **可选依赖**: ts2vec。无 ts2vec 时使用均值+标准差 fallback
