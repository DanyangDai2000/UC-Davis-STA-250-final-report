#!/usr/bin/env python
"""
ADHD 脑功能连接 · 统计迭代
=============================================
研究问题：脑功能连接（相关结构）里有没有独立于问卷的判别信号？多大？

本脚本覆盖报告 §5.3–§5.6 的四项检验：
- 去批次/去站点协变量 —— 量化 Enroll_Year/Study_Site/Scan_Location 贡献
- 偏相关（精度矩阵）替代 Pearson —— 去全局/去间接混淆
- 可解释图论特征（全局效率/聚类系数/度/特征向量中心性）—— 网络科学路径
- 中介分析 + DAG —— 问卷 → 连接模式 → 诊断

注意：数据只给了 Pearson 相关矩阵（无原始时间序列），故「部分相关」用
精度矩阵 Theta = inv(R) 近似，等价于高斯图模型下的偏相关。
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import shortest_path

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, f1_score

BASE = os.path.expanduser("~/Desktop/projects/adhd-gcn/widsdatathon2025")
OUT = os.path.expanduser("~/Desktop/projects/adhd-gcn/results")
SEED = 42
NODE = 200
N_FOLDS = 5


# ---------- 数据加载（与 train_gcn.py 一致） ----------
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


def cv_auc(X, y, model=HistGradientBoostingClassifier, **kw):
    """分层 5 折，返回 out-of-fold AUC 与加权 F1。"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pred = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = model(random_state=SEED, **kw)
        m.fit(X[tr], y[tr])
        pred[va] = m.predict_proba(X[va])[:, 1]
    hard = (pred > 0.5).astype(int)
    return roc_auc_score(y, pred), f1_score(y, hard, average="weighted")


# ---------- 图论特征 ----------
def eigenvector_centrality(B, iters=100):
    x = np.random.RandomState(0).rand(B.shape[0])
    x = x / (np.linalg.norm(x) + 1e-12)
    for _ in range(iters):
        x_new = B @ x
        n = np.linalg.norm(x_new)
        if n < 1e-12:
            break
        x_new = x_new / n
        if np.linalg.norm(x_new - x) < 1e-9:
            x = x_new
            break
        x = x_new
    return x


def graph_features(mats, topk=0.10):
    """每图提取 6 个可解释图论特征。mats: [N,200,200] Pearson。"""
    A_abs = np.abs(mats)
    N = len(mats)
    out = np.zeros((N, 6))
    for i in range(N):
        A = A_abs[i].copy()
        np.fill_diagonal(A, 0.0)
        up = A[np.triu_indices(NODE, 1)]
        thr = np.quantile(up, 1 - topk)
        B = (A >= thr).astype(float)
        B = B + B.T
        B[B > 0] = 1.0
        # 度
        k = B.sum(axis=1)
        # 聚类系数（局部，向量化）
        A3 = B @ B @ B
        tri = np.diag(A3) / 2.0
        denom = k * (k - 1)
        localC = np.where(denom > 0, 2.0 * tri / np.where(denom == 0, 1, denom), 0.0)
        C_mean = localC.mean()
        # 全局效率（距离 = 1/|corr|）
        with np.errstate(divide="ignore"):
            D = 1.0 / A
        D[~np.isfinite(D)] = 0.0
        D[A <= 0] = 0.0
        np.fill_diagonal(D, 0.0)
        sp = shortest_path(D, method="FW", directed=False)
        sp_flat = sp[np.isfinite(sp) & (sp > 0)]
        Eglob = (1.0 / sp_flat).mean() if sp_flat.size else 0.0
        # 特征向量中心性
        ec = eigenvector_centrality(B)
        # 节点强度（加权 |corr| 之和）
        strength = A.sum(axis=1)
        out[i] = [Eglob, C_mean, k.mean(), strength.mean(), ec.max(), ec.mean()]
    return out


# ---------- 偏相关（精度矩阵） ----------
def partial_corr(mats, reg=0.1):
    """从 Pearson 相关矩阵求精度矩阵 → 偏相关矩阵。"""
    P = np.zeros_like(mats)
    for i in range(len(mats)):
        R = mats[i].copy()
        np.fill_diagonal(R, 1.0)
        Rr = R + np.eye(NODE) * reg
        Theta = np.linalg.inv(Rr)
        d = np.sqrt(np.maximum(np.diag(Theta), 1e-12))
        Pi = -Theta / np.outer(d, d)
        np.fill_diagonal(Pi, 0.0)
        P[i] = Pi.astype(np.float32)
    return P


# ---------- 中介分析（线性概率模型 + bootstrap） ----------
def _ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def mediation(X, M, Y, n_boot=2000):
    a = _ols(X[:, None], M)[1]
    b = _ols(np.column_stack([X, M]), Y)[2]
    c = _ols(X[:, None], Y)[1]
    cprime = _ols(np.column_stack([X, M]), Y)[1]
    rng = np.random.default_rng(SEED)
    n = len(Y)
    boot = np.empty(n_boot)
    for j in range(n_boot):
        idx = rng.integers(0, n, n)
        ab = _ols(X[idx][:, None], M[idx])[1]
        bb = _ols(np.column_stack([X[idx], M[idx]]), Y[idx])[2]
        boot[j] = ab * bb
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(a=a, b=b, c=c, c_prime=cprime, indirect=a * b,
                ci_lo=lo, ci_hi=hi, boot=boot)


def main():
    t0 = time.time()
    ids, mats = load_connectome()
    meta = load_metadata()
    sol = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAINING_SOLUTIONS.xlsx"))
    y = sol.set_index("participant_id").loc[ids, "ADHD_Outcome"].to_numpy()
    X_tab, tab_names = build_tabular(meta, ids)
    print(f"[load] N={len(ids)} 阳性={int(y.sum())}  元数据维={X_tab.shape}  ({time.time()-t0:.0f}s)")

    # 打印元数据列名（供确认性别/年龄列）
    print("\n[meta 列名]")
    print([c for c in meta.columns if 'Sex' in str(c) or 'sex' in str(c)
           or 'Age' in str(c) or 'age' in str(c)][:20])

    # ================= ③ 去批次协变量 =================
    print("\n=== ③ 去批次/去站点协变量 ===")
    batch_kw = ["Enroll_Year", "Study_Site", "Scan_Location"]
    keep = [i for i, nm in enumerate(tab_names)
            if not any(k in nm for k in batch_kw)]
    dropped = [nm for nm in tab_names if any(k in nm for k in batch_kw)]
    auc_full, f1_full = cv_auc(X_tab, y)
    auc_deb, f1_deb = cv_auc(X_tab[:, keep], y)
    print(f"完整元数据        AUC={auc_full:.4f}  wF1={f1_full:.4f}")
    print(f"去掉批次变量({len(dropped)}列) AUC={auc_deb:.4f}  wF1={f1_deb:.4f}")
    print(f"批次变量贡献      ΔAUC={auc_full-auc_deb:+.4f}")
    pd.DataFrame([{"model": "metadata_full", "auc": auc_full, "weighted_f1": f1_full},
                  {"model": "metadata_no_batch", "auc": auc_deb, "weighted_f1": f1_deb},
                  {"model": "batch_delta", "auc": auc_full - auc_deb, "weighted_f1": f1_full - f1_deb}]
                 ).to_csv(os.path.join(OUT, "debatch.csv"), index=False)

    # ================= ④ 偏相关替代 Pearson =================
    print("\n=== ④ 偏相关（精度矩阵）替代 Pearson ===")
    flat_pearson = mats.reshape(len(ids), -1)
    Pmat = partial_corr(mats)
    flat_partial = Pmat.reshape(len(ids), -1)
    # 40000 维直接跑 GBM 太慢，统一 SVD 降维到 64（与 GBM-仅连接矩阵 口径一致）
    svd_p = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(flat_pearson)
    svd_part = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(flat_partial)
    auc_p_svd = cv_auc(svd_p.astype(np.float32), y)[0]
    auc_part_svd = cv_auc(svd_part.astype(np.float32), y)[0]
    print(f"连接矩阵 Pearson (SVD64)  AUC={auc_p_svd:.4f}")
    print(f"连接矩阵 偏相关 (SVD64)  AUC={auc_part_svd:.4f}")
    pd.DataFrame([
        {"model": "pearson_svd64", "auc": auc_p_svd},
        {"model": "partial_svd64", "auc": auc_part_svd},
    ]).to_csv(os.path.join(OUT, "partial_corr.csv"), index=False)

    # ================= ⑤ 图论特征 =================
    print("\n=== ⑤ 可解释图论特征 ===")
    GF = graph_features(mats)
    gf_cols = ["global_efficiency", "clustering_coef", "mean_degree",
               "mean_strength", "eigcentrality_max", "eigcentrality_mean"]
    gf_df = pd.DataFrame(GF, columns=gf_cols)
    gf_df.to_csv(os.path.join(OUT, "graph_features_values.csv"), index=False)
    auc_gf = cv_auc(GF.astype(np.float32), y)[0]
    auc_gf_meta = cv_auc(np.hstack([X_tab, GF.astype(np.float32)]), y)[0]
    print(f"仅图论特征            AUC={auc_gf:.4f}")
    print(f"元数据 + 图论特征     AUC={auc_gf_meta:.4f}  (元数据单独≈0.802)")
    # 每个特征单独 AUC（可解释性）
    per = {}
    for j, nm in enumerate(gf_cols):
        per[nm] = cv_auc(GF[:, j:j + 1].astype(np.float32), y)[0]
    print("单特征 AUC：", {k: round(v, 4) for k, v in per.items()})
    pd.DataFrame([{"model": "graph_features_only", "auc": auc_gf},
                  {"model": "metadata+graph_features", "auc": auc_gf_meta}]
                 ).to_csv(os.path.join(OUT, "graph_features_metrics.csv"), index=False)

    # ================= ⑥ 中介分析 =================
    print("\n=== ⑥ 中介分析（SDQ → 连接模式 → ADHD）===")
    sdq = meta.loc[ids, "SDQ_SDQ_Hyperactivity"].to_numpy(dtype=float)
    X = (sdq - sdq.mean()) / (sdq.std() + 1e-12)
    M = (GF[:, 0] - GF[:, 0].mean()) / (GF[:, 0].std() + 1e-12)  # 全局效率
    Y = y.astype(float)
    res = mediation(X, M, Y, n_boot=2000)
    sig = "显著" if (res["ci_lo"] > 0 or res["ci_hi"] < 0) else "不显著"
    print(f"路径 a (SDQ→连接)      = {res['a']:+.4f}")
    print(f"路径 b (连接→ADHD|SDQ) = {res['b']:+.4f}")
    print(f"总效应 c (SDQ→ADHD)    = {res['c']:+.4f}")
    print(f"直接效应 c'            = {res['c_prime']:+.4f}")
    print(f"间接效应 a*b           = {res['indirect']:+.4f}  "
          f"95%CI=[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  ({sig})")
    print(f"中介占比 (a*b/c)       = {res['indirect']/res['c']*100 if res['c'] else 0:+.1f}%")
    pd.DataFrame([{k: res[k] for k in ["a", "b", "c", "c_prime", "indirect", "ci_lo", "ci_hi"]}]
                 ).to_csv(os.path.join(OUT, "mediation.csv"), index=False)

    print(f"\n[done] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
