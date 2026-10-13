#!/usr/bin/env python
"""
ROI 编号 → Schaefer 200 解剖名映射
=============================================
把 gcn_saliency.csv 里的 ROI 编号(0-199) 映射到 Schaefer 2018 200-parcel
（7 networks）的解剖名。数据 ROI i 对应 LUT 编号 (i+1)。

⚠️ 顺序假设：假定 WiDS 连接矩阵的 200 列按 Schaefer 官方 parcel 顺序输出
（HBN 标准预处理的常规做法）。已用 saliency 前几名做合理性检查。
"""
import os
import pandas as pd

OUT = os.path.expanduser("~/Desktop/projects/adhd-gcn/results")


def load_labels():
    labels = []
    with open(os.path.join(OUT, "schaefer200_lut.txt")) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
    assert len(labels) == 200, f"期望 200 个标签，实际 {len(labels)}"
    return labels


def main():
    labels = load_labels()
    sal = pd.read_csv(os.path.join(OUT, "gcn_saliency.csv"))
    sal["roi_name"] = sal["roi"].map(lambda i: labels[int(i)])
    sal["network"] = sal["roi_name"].map(
        lambda n: n.split("_")[2] if n.count("_") >= 3 else n)
    sal.to_csv(os.path.join(OUT, "gcn_saliency_named.csv"), index=False)

    print("=== GCN 节点显著度 前 20 脑区（含解剖名）===")
    print(sal.head(20).to_string(index=False))

    print("\n=== 显著脑区的网络分布（前 50）===")
    print(sal.head(50)["network"].value_counts().to_string())


if __name__ == "__main__":
    main()
