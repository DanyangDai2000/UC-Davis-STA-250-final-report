#!/usr/bin/env python
"""
中介分析 + DAG
=============================================
模型：SDQ_Hyperactivity(问卷) → 连接模式(图论特征·全局效率) → ADHD(诊断)
用线性概率模型（OLS 拟合二分类 Y）做中介分解，bootstrap 求间接效应 CI。

主分析：无协变量；稳健性：加年龄(MRI_Track_Age_at_Scan)协变量。
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/Desktop/projects/adhd-gcn/widsdatathon2025")
OUT = os.path.expanduser("~/Desktop/projects/adhd-gcn/results")
SEED = 42


def load_connectome_ids():
    path = os.path.join(BASE, "TRAIN_NEW",
                        "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
    return pd.read_csv(path)["participant_id"].tolist()


def load_metadata():
    q = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAIN_QUANTITATIVE_METADATA_new.xlsx"))
    c = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAIN_CATEGORICAL_METADATA_new.xlsx"))
    return q.merge(c, on="participant_id", how="left").set_index("participant_id")


def _ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def mediation(X, M, Y, C=None, n_boot=5000, seed=SEED):
    """X,M 各一列；Y 二值；C 可选协变量矩阵。返回路径系数与间接效应 CI。"""
    # 全样本路径
    XM = np.column_stack([X, M])
    XM = np.hstack([XM, C]) if C is not None else XM
    a = _ols(X, M)[1]                       # X -> M
    b = _ols(XM, Y)[2]                      # M -> Y | X(,C)
    c = _ols(X, Y)[1]                       # 总效应 X -> Y
    cprime = _ols(XM, Y)[1]                 # 直接效应 X -> Y | M(,C)
    rng = np.random.default_rng(seed)
    n = len(Y)
    boot = np.empty(n_boot)
    for j in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]
        ab = _ols(Xb, Mb)[1]
        XMb = np.column_stack([Xb, Mb])
        if C is not None:
            XMb = np.hstack([XMb, C[idx]])
        bb = _ols(XMb, Yb)[2]
        boot[j] = ab * bb
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(a=a, b=b, c=c, c_prime=cprime, indirect=a * b,
                ci_lo=lo, ci_hi=hi)


def main():
    ids = load_connectome_ids()
    meta = load_metadata()
    sol = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAINING_SOLUTIONS.xlsx"))
    y = sol.set_index("participant_id").loc[ids, "ADHD_Outcome"].to_numpy(dtype=float)

    GF = pd.read_csv(os.path.join(OUT, "graph_features_values.csv")).to_numpy()
    assert not np.isnan(GF).any(), "图论特征含 NaN"

    sdq = meta.loc[ids, "SDQ_SDQ_Hyperactivity"].to_numpy(dtype=float)
    print(f"SDQ 缺失数 = {int(np.isnan(sdq).sum())}")
    sdq = pd.Series(sdq).fillna(pd.Series(sdq).median()).to_numpy()

    X = (sdq - sdq.mean()) / (sdq.std() + 1e-12)
    M = (GF[:, 0] - GF[:, 0].mean()) / (GF[:, 0].std() + 1e-12)  # 全局效率
    Y = y

    # 年龄协变量
    age = meta.loc[ids, "MRI_Track_Age_at_Scan"].to_numpy(dtype=float)
    age = pd.Series(age).fillna(pd.Series(age).median()).to_numpy()
    C = ((age - age.mean()) / (age.std() + 1e-12))[:, None]

    print("\n=== ⑥ 中介分析：SDQ → 全局效率(连接模式) → ADHD ===")
    for label, Cov in [("无协变量", None), ("加年龄协变量", C)]:
        r = mediation(X, M, Y, C=Cov, n_boot=5000)
        sig = "显著" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "不显著"
        prop = (r["indirect"] / r["c"] * 100) if r["c"] else 0.0
        print(f"\n[{label}]")
        print(f"  a (SDQ→连接)       = {r['a']:+.4f}")
        print(f"  b (连接→ADHD|SDQ)  = {r['b']:+.4f}")
        print(f"  c (总效应 SDQ→ADHD) = {r['c']:+.4f}")
        print(f"  c'(直接效应)        = {r['c_prime']:+.4f}")
        print(f"  间接效应 a*b        = {r['indirect']:+.4f}  "
              f"95%CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  ({sig})")
        print(f"  中介占比 a*b/c      = {prop:+.1f}%")
        if label == "无协变量":
            main_res = r
            main_res["label"] = "no_covariate"
    main_res["label"] = "no_covariate"
    pd.DataFrame([main_res]).to_csv(os.path.join(OUT, "mediation.csv"), index=False)


if __name__ == "__main__":
    main()
