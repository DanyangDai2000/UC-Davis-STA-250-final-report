# ADHD 脑功能连接中的独立判别信号

基于 **WiDS Datathon 2025** 的 fMRI 功能连接数据，回答一个统计问题：
**脑功能连接（相关结构）里到底有没有独立于问卷的判别信号？如果有，多大？**
方法上用图卷积网络（GCN）配合多层对照基线，并规范评估（subject-level 切分、
加权 F1、水位基线、SHAP 归因、bootstrap 显著性、中介分析）。

## 数据

- 来源：WiDS Datathon 2025（HBN 儿童脑影像队列）
- 训练样本：**1,213** 个受试者，标签 `ADHD_Outcome`（1=ADHD **831 / 68.5%**，0=对照 382 / 31.5%）
- 每个受试者一张 **200 × 200 脑区 Pearson 功能连接矩阵**（上三角 19900 维，`..._36P_Pearson.csv`）
- 另有定量元数据（EHQ / 色觉 / APQ 教养问卷 / SDQ 行为问卷 / 年龄）与分类元数据（站点 / 种族 / 扫描地 / Barratt 社会经济地位）

## 方法

把每个受试者建模成一张**固定 200 节点的图**：

- **节点**：200 个脑区（ROI）
- **边**：脑区间功能连接，权重 = `|Pearson 相关系数|`，加自环后对称归一化 `D^{-1/2} A D^{-1/2}`
- **节点特征**：该脑区到所有脑区的连接指纹（即连接矩阵的每一行）

模型：**两层 GCN + ReLU + 全局平均池化 + 二分类 logits**（`torch` 手写，无依赖 torch-geometric）。

### 评估口径

1. **subject-level 切分**：每个受试者只有 1 行矩阵，用分层 5 折交叉验证，同一受试者绝不跨 train/test，杜绝泄漏。
2. **主指标 = 加权 F1**，另报每类 F1、ROC-AUC、balanced accuracy、accuracy。
3. **水位基线**：① 多数类基线 ② 逻辑回归（元数据） ③ 梯度提升（元数据）——看连接矩阵 / GCN 相对「只靠问卷」能加多少。
4. **SHAP 归因**：对梯度提升基线跑 `TreeExplainer`；对 GCN 算**梯度 × 输入**的节点显著度（哪些脑区最被看重）。

## 运行

```bash
# 依赖（托管 venv）：torch, shap, scikit-learn, scipy, pandas, openpyxl
python code/train_gcn.py
```

输出到 `results/`：`metrics.csv` · `shap_top.csv` · `gcn_saliency.csv`。

## 结果

5 折分层交叉验证（subject-level，无泄漏），主指标 = 加权 F1：

| model | accuracy | weighted F1 | macro F1 | F1(对照) | F1(ADHD) | ROC-AUC | balanced acc |
|---|--:|--:|--:|--:|--:|--:|--:|
| majority baseline | 0.685 | 0.557 | 0.407 | 0.000 | 0.813 | 0.500 | 0.500 |
| logistic（元数据） | 0.760 | 0.767 | 0.739 | 0.665 | 0.813 | 0.840 | 0.759 |
| gradient boosting（元数据） | 0.787 | 0.781 | 0.740 | 0.630 | 0.850 | 0.802 | 0.730 |
| gradient boosting（仅连接矩阵） | 0.645 | 0.574 | 0.455 | 0.133 | 0.777 | 0.505 | 0.494 |
| gradient boosting（元数据＋连接矩阵） | 0.791 | 0.782 | 0.738 | 0.620 | 0.856 | 0.819 | 0.723 |
| **GCN（仅连接矩阵）** | 0.580 | 0.587 | 0.531 | 0.379 | 0.683 | 0.555 | 0.533 |

### 结论

1. **行为问卷是主信号**：SHAP 排名第一的 `SDQ_Hyperactivity`（mean|SHAP|≈3.0）远超其余特征；只靠问卷的逻辑回归就有 **ROC-AUC 0.84**。
2. **功能连接矩阵单独几乎无信号**：仅用连接矩阵，GCN AUC 0.555、梯度提升 AUC 0.505，都接近随机 —— 单独看脑连接图分不出 ADHD。
3. **连接矩阵只有少量增量**：元数据 GBM 从 0.802 → 加连接矩阵 0.819（＋0.017）；GCN 相对同源扁平基线（0.505）略高（0.555），说明**图结构能多榨出一点信号，但远不足以单独支撑分类**。

> 这份工作的价值在于：用 subject-level 切分 ＋ 加权 F1 ＋ 水位基线 ＋ SHAP，把「连接矩阵到底有没有独立信号」讲清楚 —— 结论是「有、但很小」，而不是空口说「建了个 GCN」。

## 文件结构

```
adhd-gcn/
├── code/train_gcn.py              # 主结果：GCN + 基线 + 评估 + SHAP
├── code/iterate_statistics.py     # 去批次 · 偏相关 · 图论特征（§5.3–5.5）
├── code/mediation.py              # 中介分析 + DAG（§5.6）
├── code/bootstrap_significance.py # GCN vs 扁平 显著性检验（§5.1）
├── code/roi_labels.py             # ROI → Schaefer 解剖名（§5.2）
├── inspect_data.py                # 数据字典 / 标签分布 / 矩阵形状探查
├── report/report.md               # 统计报告：相关性与因果性的边界
├── results/                       # metrics·shap_top·gcn_saliency_named·debatch·partial_corr·graph_features·mediation·bootstrap
└── widsdatathon2025/              # 原始数据（不进 git）
```

完整统计报告见 [`report/report.md`](report/report.md)：在关联结果基础上，
重点辨析**相关性与因果性**——功能连接是相关结构而非因果图、预测力强不等于
因果、SDQ 问卷主导背后的「构造性相关」，以及六项统计迭代（§5）把「连接
矩阵到底有多少独立信号」追到底。
