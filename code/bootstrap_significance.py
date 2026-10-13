#!/usr/bin/env python
"""
bootstrap / 多种子显著性检验
=============================================
问题：「GCN(仅连接矩阵) AUC≈0.555」 与 「扁平 SVD-GBM(仅连接矩阵) AUC≈0.505」
的差距是否稳定显著 —— 即「图结构」相对「扁平特征」到底有没有真实增量。

做法：N_SEEDS 个不同随机种子，每个种子做一次 subject-level 分层 5 折，
分别跑 GCN 与扁平 SVD-GBM，得到成对的 out-of-fold AUC。
对成对 AUC 做 Wilcoxon 符号秩检验 + 配对 t 检验，并给差值的 bootstrap CI。
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from train_gcn import (load_connectome, load_metadata, sparse_adjacency,
                       node_features, GCN, train_gcn, BASE, OUT, NODE)

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, ttest_rel

N_SEEDS = 10
EPOCHS = 100
N_FOLDS = 5


def run_once(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ids, mats = load_connectome()
    meta = load_metadata()
    sol = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAINING_SOLUTIONS.xlsx"))
    y = sol.set_index("participant_id").loc[ids, "ADHD_Outcome"].to_numpy()

    A_hat = sparse_adjacency(mats)
    X_feat = node_features(mats)
    flat = mats.reshape(len(ids), -1)
    svd = TruncatedSVD(n_components=64, random_state=seed).fit_transform(flat)

    pos_weight = (1 - y.mean()) / y.mean()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    gcn_pred = np.zeros(len(y))
    gb_pred = np.zeros(len(y))
    for tr, va in skf.split(ids, y):
        _, p = train_gcn(X_feat[tr], A_hat[tr], y[tr],
                         X_feat[va], A_hat[va], y[va], pos_weight, epochs=EPOCHS)
        gcn_pred[va] = p
        gb = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
        gb.fit(svd[tr].astype(np.float32), y[tr])
        gb_pred[va] = gb.predict_proba(svd[va].astype(np.float32))[:, 1]
    return roc_auc_score(y, gcn_pred), roc_auc_score(y, gb_pred)


def main():
    t0 = time.time()
    rows = []
    for s in range(N_SEEDS):
        seed = 100 + s
        a_gcn, a_gb = run_once(seed)
        rows.append({"seed": seed, "gcn_auc": a_gcn, "flat_auc": a_gb})
        print(f"seed={seed}  GCN={a_gcn:.4f}  扁平={a_gb:.4f}  "
              f"Δ={a_gcn-a_gb:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    diff = df["gcn_auc"] - df["flat_auc"]
    w, wp = wilcoxon(df["gcn_auc"], df["flat_auc"])
    t, tp = ttest_rel(df["gcn_auc"], df["flat_auc"])

    # bootstrap CI of mean difference
    rng = np.random.default_rng(0)
    boots = [rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean()
             for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print("\n=== 结果 ===")
    print(df.to_string(index=False))
    print(f"\nGCN 均值 AUC      = {df['gcn_auc'].mean():.4f} ± {df['gcn_auc'].std():.4f}")
    print(f"扁平均值 AUC      = {df['flat_auc'].mean():.4f} ± {df['flat_auc'].std():.4f}")
    print(f"差值均值          = {diff.mean():+.4f}  (bootstrap 95%CI=[{lo:+.4f}, {hi:+.4f}])")
    print(f"Wilcoxon 符号秩   W={w:.1f}  p={wp:.4f}")
    print(f"配对 t 检验       t={t:.3f}  p={tp:.4f}")

    out = df.copy()
    out["diff"] = diff
    out.to_csv(os.path.join(OUT, "bootstrap_gcn_vs_flat.csv"), index=False)
    summary = pd.DataFrame([{
        "gcn_auc_mean": df["gcn_auc"].mean(), "gcn_auc_sd": df["gcn_auc"].std(),
        "flat_auc_mean": df["flat_auc"].mean(), "flat_auc_sd": df["flat_auc"].std(),
        "diff_mean": diff.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
        "wilcoxon_p": wp, "ttest_p": tp,
    }])
    summary.to_csv(os.path.join(OUT, "bootstrap_summary.csv"), index=False)
    print(f"\n[done] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
