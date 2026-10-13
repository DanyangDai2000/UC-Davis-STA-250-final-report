#!/usr/bin/env python
"""
ADHD 脑功能连接中的独立判别信号 · 图卷积网络（GCN）
====================================================
数据：WiDS Datathon 2025（fMRI 功能连接矩阵）
研究问题：脑功能连接（相关结构）里有没有独立于问卷的判别信号？多大？

评估口径：
1. subject-level 切分 —— 每受试者 1 行矩阵，分层 5 折，无泄漏。
2. 主指标 = 加权 F1；另报每类 F1 / ROC-AUC / balanced accuracy。
3. 水位基线：多数类 + 逻辑回归(元数据) + 梯度提升(元数据)
   + 梯度提升(元数据 + 连接矩阵 SVD)，定位「连接矩阵」本身有没有增量。
4. SHAP 归因：GB 基线 TreeExplainer；GCN 梯度×输入节点显著度。

模型：
- 邻接稀疏化（每图只保留 |corr| 最强的 top 10% 边），避免稠密图过平滑。
- 节点特征 = 连接指纹(200) + 节点强度(1)。
- 训练：200 epoch、weight_decay。
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score)

BASE = os.path.expanduser("~/Desktop/projects/adhd-gcn/widsdatathon2025")
OUT = os.path.expanduser("~/Desktop/projects/adhd-gcn/results")
SEED = 42
N_FOLDS = 5
NODE = 200

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cpu"


def load_connectome():
    path = os.path.join(BASE, "TRAIN_NEW",
                        "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
    df = pd.read_csv(path)
    ids = df["participant_id"].tolist()
    vals = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    mats = np.zeros((len(ids), NODE, NODE), dtype=np.float32)
    for i in range(len(ids)):
        mats[i] = squareform(vals[i]).astype(np.float32)
    return ids, mats


def load_metadata():
    q = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAIN_QUANTITATIVE_METADATA_new.xlsx"))
    c = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAIN_CATEGORICAL_METADATA_new.xlsx"))
    df = q.merge(c, on="participant_id", how="left").set_index("participant_id")
    return df


def build_tabular(meta, ids):
    num_cols = [c for c in meta.columns if meta[c].dtype != object and c != "participant_id"]
    cat_cols = [c for c in meta.columns if meta[c].dtype == object]
    X = meta.loc[ids, num_cols].copy().apply(pd.to_numeric, errors="coerce")
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    parts = [X.to_numpy(dtype=np.float32)]
    names = list(num_cols)
    for c in cat_cols:
        d = pd.get_dummies(meta.loc[ids, c], prefix=c, dummy_na=True)
        parts.append(d.to_numpy(dtype=np.float32))
        names += list(d.columns)
    return np.hstack(parts), names


def sparse_adjacency(mats, topk_frac=0.10):
    """稀疏化 + 对称归一化邻接：保留 |corr| 最强 topk 边，加自环，D^-1/2 A D^-1/2。"""
    A = np.abs(mats).astype(np.float32)
    n = A.shape[0]
    Aout = np.zeros_like(A)
    for i in range(n):
        a = A[i].copy()
        np.fill_diagonal(a, 0.0)
        up = a[np.triu_indices(NODE, 1)]
        thr = np.quantile(up, 1 - topk_frac) if up.size else 0.0
        a[a < thr] = 0.0
        a = a + a.T
        Aout[i] = a
    Aout += np.eye(NODE, dtype=np.float32)
    d = Aout.sum(axis=2, keepdims=True) ** -0.5
    return (d * Aout * d.transpose(0, 2, 1)).astype(np.float32)


def node_features(mats):
    """节点特征 = [连接指纹 200 | 节点强度 1] -> [N, 200, 201]。"""
    strength = np.abs(mats).sum(axis=2, keepdims=True)  # [N,200,1]
    return np.concatenate([mats, strength], axis=2).astype(np.float32)


class GCN(nn.Module):
    def __init__(self, in_dim=201, hidden=128, dropout=0.4):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(in_dim, hidden))
        self.w2 = nn.Parameter(torch.empty(hidden, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.b2 = nn.Parameter(torch.zeros(hidden))
        self.w_out = nn.Parameter(torch.empty(hidden, 1))
        self.b_out = nn.Parameter(torch.zeros(1))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w_out)

    def forward(self, X, A_hat):
        h = F.relu(A_hat @ X @ self.w1 + self.b1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(A_hat @ h @ self.w2 + self.b2)
        h = h.mean(dim=1)
        return h @ self.w_out + self.b_out


def train_gcn(X_tr, A_tr, y_tr, X_va, A_va, y_va, pos_weight, epochs=200, lr=0.01):
    Xtr = torch.tensor(X_tr); Atr = torch.tensor(A_tr); ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xva = torch.tensor(X_va); Ava = torch.tensor(A_va)
    model = GCN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr, Atr).squeeze(1)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xva, Ava).squeeze(1)).numpy()
    return model, p


def gcn_saliency(model, X, A):
    model.eval()
    x = torch.tensor(X, requires_grad=True)
    a = torch.tensor(A)
    logits = model(x, a).squeeze(0)
    logits.backward()
    sal = (x.grad.detach() * x.detach()).abs().sum(dim=2).squeeze(0).numpy()
    return sal


def main():
    ids, mats = load_connectome()
    meta = load_metadata()
    sol = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAINING_SOLUTIONS.xlsx"))
    y = sol.set_index("participant_id").loc[ids, "ADHD_Outcome"].to_numpy()
    print(f"样本 {len(ids)} · 阳性 {int(y.sum())} ({y.mean()*100:.1f}%)")

    A_hat = sparse_adjacency(mats)
    X_feat = node_features(mats)
    X_tab, tab_names = build_tabular(meta, ids)

    # 连接矩阵 SVD 特征（用于「连接矩阵有没有信号」的表格基线）
    flat = mats.reshape(len(ids), -1)
    svd = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(flat)
    X_conn = svd.astype(np.float32)
    X_tab_conn = np.hstack([X_tab, X_conn])

    pos_weight = (1 - y.mean()) / y.mean()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(ids, y))

    preds = {k: np.zeros(len(y)) for k in
             ["gcn", "lr", "gb_meta", "gb_conn", "gb_meta_conn"]}
    t0 = time.time()
    gcn_model = None
    for fold, (tr, va) in enumerate(folds):
        gcn_model, p = train_gcn(X_feat[tr], A_hat[tr], y[tr],
                                 X_feat[va], A_hat[va], y[va], pos_weight)
        preds["gcn"][va] = p

        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_tab[tr], y[tr])
        preds["lr"][va] = lr.predict_proba(X_tab[va])[:, 1]

        gb = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
        gb.fit(X_tab[tr], y[tr])
        preds["gb_meta"][va] = gb.predict_proba(X_tab[va])[:, 1]

        gbo = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
        gbo.fit(X_conn[tr], y[tr])
        preds["gb_conn"][va] = gbo.predict_proba(X_conn[va])[:, 1]

        gbc = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
        gbc.fit(X_tab_conn[tr], y[tr])
        preds["gb_meta_conn"][va] = gbc.predict_proba(X_tab_conn[va])[:, 1]
        print(f"fold {fold+1}/{N_FOLDS} done ({time.time()-t0:.0f}s)")

    maj = np.full(len(y), 1)
    rows = []
    def record(name, p):
        hard = (p > 0.5).astype(int)
        rows.append({
            "model": name, "accuracy": accuracy_score(y, hard),
            "weighted_f1": f1_score(y, hard, average="weighted"),
            "macro_f1": f1_score(y, hard, average="macro"),
            "f1_class0": f1_score(y, hard, pos_label=0),
            "f1_class1": f1_score(y, hard, pos_label=1),
            "roc_auc": roc_auc_score(y, p),
            "balanced_acc": balanced_accuracy_score(y, hard),
        })
    record("majority_baseline", maj)
    record("logistic_metadata", preds["lr"])
    record("gradientboosting_metadata", preds["gb_meta"])
    record("gradientboosting_connectome_only", preds["gb_conn"])
    record("gradientboosting_metadata+connectome", preds["gb_meta_conn"])
    record("GCN_connectome", preds["gcn"])

    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    print("\n" + mdf.round(4).to_string(index=False))

    # SHAP
    import shap
    gb_full = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
    gb_full.fit(X_tab, y)
    sv = shap.TreeExplainer(gb_full).shap_values(X_tab)
    if isinstance(sv, list):
        sv = sv[1]
    imp = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({"feature": tab_names, "mean_abs_shap": imp})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(os.path.join(OUT, "shap_top.csv"), index=False)
    print("\nSHAP 前 12 特征:")
    print(shap_df.head(12).round(3).to_string(index=False))

    # GCN 节点显著度
    sal_avg = np.zeros(NODE)
    K = min(200, len(ids))
    for i in range(K):
        sal_avg += gcn_saliency(gcn_model, X_feat[i:i+1], A_hat[i:i+1])
    sal_avg /= K
    sal_df = pd.DataFrame({"roi": np.arange(NODE), "saliency": sal_avg})
    sal_df = sal_df.sort_values("saliency", ascending=False)
    sal_df.to_csv(os.path.join(OUT, "gcn_saliency.csv"), index=False)
    print("\nGCN 节点显著度前 10 脑区:", sal_df["roi"].head(10).tolist())


if __name__ == "__main__":
    main()
