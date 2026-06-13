# Generation Metrics

## Point Metrics (样本级指标)

比较真实样本集合与生成样本集合的统计特性。

**输入形状**:
- `real`: `(N, C, T)` — N 条真实样本
- `generated`: `(M, C, T)` — M 条生成样本（M 可以不等于 N）

### Fidelity (保真度)

$$\text{Fidelity} = \frac{1}{C \cdot T} \sum_{i=1}^{C \cdot T} \frac{1}{L} \sum_{j=1}^{L} |x^{\text{real}}_{(j),i} - x^{\text{gen}}_{(j),i}|$$

其中 $x_{(j)}$ 是第 $j$ 小的值（排序后），$L = \min(N, M)$。

- **含义**: 1D Wasserstein 距离，衡量真实与生成样本的边际分布差异
- **越小越好**，0 表示分布完全一致
- **计算方法**: 对每个特征-时间步位置，分别对真实和生成样本排序，计算排序后的绝对差均值

### Discriminative Score (判别分数)

$$\text{DS} = |\text{Accuracy} - 0.5|$$

训练一个线性分类器区分真实样本与生成样本，Accuracy 是分类准确率。

- **含义**: 衡量生成样本与真实样本的可区分度
- **范围**: $[0, 0.5]$
- **越小越好**，0 表示完美不可区分（生成质量高）
- **实现**: 80/20 训练/测试划分，100 步 Adam 优化

### Correlation (自相关结构)

$$\text{Corr} = \frac{1}{|\mathcal{L}|} \sum_{\text{lag} \in \mathcal{L}} |\text{ACF}^{\text{real}}(\text{lag}) - \text{ACF}^{\text{gen}}(\text{lag})|$$

其中 $\text{ACF}(\text{lag})$ 是滞后 lag 的自相关系数。

- **含义**: 衡量生成样本是否保留了真实样本的时间自相关结构
- **越小越好**，0 表示自相关结构完全一致
- **参考**: Diffusion-TS (ICLR 2024)

### KL Divergence (KL 散度)

$$\text{KL} = \frac{1}{C} \sum_{c=1}^{C} \sum_{k=1}^{K} p_{c,k} \log \frac{p_{c,k}}{q_{c,k}}$$

其中 $p_{c,k}$ 和 $q_{c,k}$ 是第 $c$ 个特征的第 $k$ 个直方图 bin 的概率（真实 vs 生成）。

- **含义**: 真实分布与生成分布的 KL 散度（非对称）
- **越小越好**，0 表示分布完全一致
- **实现**: 50 个 bins 的直方图近似，加 $10^{-8}$ 平滑

---

## Probabilistic Metrics (分布级指标)

### MMD (Maximum Mean Discrepancy)

$$\text{MMD}^2 = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]$$

其中 $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$ 是 RBF 核，$\sigma$ 取中位数距离。

- **含义**: 最大均值差异，衡量两个分布在再生核希尔伯特空间中的距离
- **越小越好**，0 表示分布完全一致
- **优势**: 不依赖密度估计，直接比较样本

### JS Divergence (Jensen-Shannon 散度)

$$\text{JS} = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M)$$

其中 $M = \frac{1}{2}(P + Q)$，$P$ 和 $Q$ 是真实与生成的边际分布。

- **含义**: 对称且有界的分布距离
- **范围**: $[0, \log 2]$
- **越小越好**，0 表示分布完全一致
- **实现**: 50 个 bins 的直方图近似

### Log-Likelihood (对数似然)

$$\text{LL} = \frac{1}{N \cdot C \cdot T} \sum_{b,c,t} -\frac{1}{2} \left( \log(2\pi \sigma^2_{c,t}) + \frac{(x^{\text{real}}_{b,c,t} - \mu_{c,t})^2}{\sigma^2_{c,t}} \right)$$

其中 $\mu_{c,t}$ 和 $\sigma^2_{c,t}$ 是生成样本的均值和方差。

- **含义**: 真实数据在生成数据拟合的高斯分布下的对数似然
- **越大越好**
- **假设**: 生成数据服从高斯分布

---

## 与 Prediction/Imputation 的区别

| 任务 | 比较对象 | 典型场景 |
|------|---------|---------|
| **Prediction** | 逐点对比：`forecast` vs `target` | 预测未来值 |
| **Imputation** | 逐点对比：`imputed` vs `target`（带 mask） | 填补缺失值 |
| **Generation** | 分布对比：`generated` 集合 vs `real` 集合 | 生成新样本 |

Generation 指标关注**分布级别**的相似性，而非逐点误差。
