"""
量測 GT mask 的實際 line pitch / run length，用來決定 V4 的 threshold_window。

V4 的 T = LocalMean(I, w) + dT。若 w 小於 pitch，local mean 會被線條本身帶走，
等於沒有參考基準；w 應該涵蓋至少一個完整週期（line + space）。
"""

import json
from collections import defaultdict

import numpy as np

from src.data import P2PDataset, load_pairs

pairs = load_pairs("pairs.json")
split = json.load(open("splits/split_seed0.json", encoding="utf-8"))
ds = P2PDataset(split["train"], pairs, mode="full", augment=False)

by_fam = defaultdict(list)
for i in range(len(ds)):
    b = ds[i]
    m = b["y"][0].numpy() > 0.5
    row = m[m.shape[0] // 2]                      # 取中間一列
    # run-length：連續同值的長度
    change = np.flatnonzero(np.diff(row.astype(np.int8))) + 1
    if len(change) < 2:
        continue
    runs = np.diff(np.concatenate([[0], change, [len(row)]]))
    runs = runs[1:-1]                             # 去掉不完整的頭尾
    if len(runs) >= 2:
        by_fam[b["family"]].append((np.median(runs), np.median(runs[:-1] + runs[1:])))

print(f"{'family':<10} {'n':>4} {'median run (px)':>16} {'median pitch (px)':>18}")
print("-" * 52)
all_pitch = []
for fam in sorted(by_fam):
    v = np.array(by_fam[fam])
    print(f"{fam:<10} {len(v):>4} {np.median(v[:,0]):>16.1f} {np.median(v[:,1]):>18.1f}")
    all_pitch.extend(v[:, 1])

all_pitch = np.array(all_pitch)
print("-" * 52)
print(f"整體 pitch: median={np.median(all_pitch):.1f}  "
      f"p90={np.percentile(all_pitch,90):.1f}  max={all_pitch.max():.1f}")

w = int(np.percentile(all_pitch, 90)) * 2 + 1
w = w if w % 2 else w + 1
print(f"\n=> 建議 threshold_window ≈ {w}（涵蓋 p90 pitch 的兩倍，取奇數）")
print("   若記憶體吃緊可用 median 的兩倍；太小會讓 local mean 被線條帶走。")
