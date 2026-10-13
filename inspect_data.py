#!/usr/bin/env python
"""探查 WiDS 2025 ADHD 数据集结构与标签分布。"""
import os
import pandas as pd

BASE = os.path.expanduser("~/Desktop/projects/adhd-gcn/widsdatathon2025")

print("=" * 70)
print("1) 数据字典 (Data Dictionary.xlsx)")
print("=" * 70)
dd = pd.ExcelFile(os.path.join(BASE, "Data Dictionary.xlsx"))
print("sheets:", dd.sheet_names)
for s in dd.sheet_names:
    df = dd.parse(s)
    print(f"\n--- sheet: {s}  shape={df.shape} ---")
    print(df.head(20).to_string())

print("\n" + "=" * 70)
print("2) 训练标签 (TRAINING_SOLUTIONS.xlsx)")
print("=" * 70)
sol = pd.read_excel(os.path.join(BASE, "TRAIN_NEW", "TRAINING_SOLUTIONS.xlsx"))
print("shape:", sol.shape)
print("columns:", list(sol.columns))
print(sol.head(10).to_string())
if sol.shape[1] >= 2:
    label_col = sol.columns[-1]
    print("\n标签分布:")
    print(sol[label_col].value_counts(dropna=False).to_string())

print("\n" + "=" * 70)
print("3) 功能连接矩阵 (TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv)")
print("=" * 70)
mfile = os.path.join(BASE, "TRAIN_NEW", "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
with open(mfile) as f:
    header = f.readline().strip()
    cols = header.split(",")
    print("总列数:", len(cols))
    print("前 8 列名:", cols[:8])
    print("后 4 列名:", cols[-4:])
    for i in range(4):
        line = f.readline().strip()
        parts = line.split(",")
        print(f"第{i+1}行: 列数={len(parts)} 前5值={parts[:5]} 尾2值={parts[-2:]}")

print("\n" + "=" * 70)
print("4) 元数据")
print("=" * 70)
for f in ["TRAIN_QUANTITATIVE_METADATA_new.xlsx", "TRAIN_CATEGORICAL_METADATA_new.xlsx"]:
    p = os.path.join(BASE, "TRAIN_NEW", f)
    df = pd.read_excel(p)
    print(f"\n--- {f}  shape={df.shape} ---")
    print("columns:", list(df.columns))
    print(df.head(5).to_string())

print("\n" + "=" * 70)
print("5) TRAIN_OLD 与 TEST 概览")
print("=" * 70)
for f in ["TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"]:
    p = os.path.join(BASE, "TRAIN_OLD", f)
    with open(p) as fh:
        h = fh.readline().strip()
        print(f"TRAIN_OLD/{f}: 列数={len(h.split(','))}")
for f in ["TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"]:
    p = os.path.join(BASE, "TEST", f)
    with open(p) as fh:
        h = fh.readline().strip()
        print(f"TEST/{f}: 列数={len(h.split(','))}")
