"""
檢查 dataset/ 的結構與影像屬性。

    python -m src.inspect_data --root dataset

配對不分大小寫（input/Proximity_N 對 gt/proximity_N 這種情況），
與 src/data.py 的 scan_pairs 行為一致。
只需要 pillow + numpy，不需要 torch。
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def index_dir(root: Path) -> dict[str, str]:
    """{相對路徑小寫: 實際相對路徑}"""
    idx = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                rel = (Path(d) / f).relative_to(root).as_posix()
                idx[rel.lower()] = rel
    return idx


def describe(p: Path) -> dict:
    im = Image.open(p)
    a = np.array(im.convert("L"))
    u = np.unique(a)
    # 二值圖存成 JPEG 時邊緣會多出少量中間灰階；看「中間灰階的像素比例」
    # 比看 unique 值個數可靠。
    mid = float(((a > 32) & (a < 223)).mean())
    return {"size": im.size, "mode": im.mode, "n_unique": int(u.size),
            "min": int(a.min()), "max": int(a.max()), "mid_ratio": mid,
            "binary": u.size <= 2 or mid < 0.02, "fg": float((a > 127).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--input-dir", default="input")
    ap.add_argument("--gt-dir", default="gt")
    ap.add_argument("--n", type=int, default=3, help="每個 class 取樣幾張")
    args = ap.parse_args()

    root = Path(args.root)
    in_root, gt_root = root / args.input_dir, root / args.gt_dir
    for d in (in_root, gt_root):
        if not d.exists():
            raise SystemExit(f"找不到 {d}")

    idx_in, idx_gt = index_dir(in_root), index_dir(gt_root)

    paired, case_diff, missing = [], [], []
    for k, rel in sorted(idx_in.items()):
        if k in idx_gt:
            paired.append((rel, idx_gt[k]))
            if idx_gt[k] != rel:
                case_diff.append((rel, idx_gt[k]))
        else:
            missing.append(rel)
    orphan = [v for k, v in idx_gt.items() if k not in idx_in]

    print(f"=== {root} ===")
    print(f"input : {len(idx_in)} files")
    print(f"gt    : {len(idx_gt)} files")
    print(f"配對（不分大小寫）: {len(paired)}")
    if case_diff:
        print(f"  大小寫不一致但已配對: {len(case_diff)} 筆")
        for a, b in case_diff[:2]:
            print(f"      {a}  <->  {b}")
    if missing:
        print(f"  ** input 有但 gt 沒有 {len(missing)} 筆: {missing[:3]}")
    if orphan:
        print(f"  ** gt 有但 input 沒有 {len(orphan)} 筆: {orphan[:3]}")
    if not missing and not orphan:
        print("  -> 完全對應")

    # ---------- class 結構 ----------
    by_cls = defaultdict(list)
    for rel, _ in paired:
        by_cls[rel.split("/")[0] if "/" in rel else "(flat)"].append(rel)

    print(f"\n=== class 結構 ===")
    print(f"第一層子資料夾共 {len(by_cls)} 個")
    if len(by_cls) == 12:
        print("  -> 與 slide 8 的 12 class 吻合")
    else:
        print(f"  -> slide 8 說 12 class，這裡是 {len(by_cls)} 個，要確認")

    # ---------- 取樣 + per-class 彙總 ----------
    print(f"\n=== 每個 class 取樣 {args.n} 張 ===")
    print(f"{'class':<16}{'n':>6}{'gt 前景佔比':>14}   {'範例':<28}")
    print("-" * 72)
    sizes_in, sizes_gt, all_binary, rows = set(), set(), [], []
    gt_map = dict(paired)
    for c, rels in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
        fgs = []
        for rel in rels[:args.n]:
            di = describe(in_root / rel)
            dg = describe(gt_root / gt_map[rel])
            sizes_in.add(di["size"]); sizes_gt.add(dg["size"])
            all_binary.append(dg["binary"])
            fgs.append(dg["fg"])
        rows.append((c, len(rels), float(np.mean(fgs))))
        print(f"{c:<16}{len(rels):>6}{np.mean(fgs):>14.3f}   {Path(rels[0]).name:<28}")

    # ---------- 判讀 ----------
    print(f"\n=== 判讀 ===")
    print(f"  input 尺寸: {sorted(sizes_in)}")
    print(f"  gt    尺寸: {sorted(sizes_gt)}")
    if sizes_in != sizes_gt:
        print("  ** input/gt 尺寸不同")
    if sizes_in != {(512, 512)}:
        print("  ** 不是 512x512：resize 會改變 EPE / CD 的像素單位")
    print(f"  gt 全部二值（容許 JPEG 壓縮雜訊）: {all(all_binary)}")

    fg = np.array([r[2] for r in rows])
    print(f"  gt 前景佔比: {fg.min():.3f} ~ {fg.max():.3f}")
    if fg.max() - fg.min() > 0.3:
        hi = [r[0] for r in rows if r[2] > 0.5]
        lo = [r[0] for r in rows if r[2] <= 0.5]
        print("  ** 前景佔比跨度很大，資料含兩種極性：")
        print(f"       高前景 ({len(hi)} class): {', '.join(sorted(hi))}")
        print(f"       低前景 ({len(lo)} class): {', '.join(sorted(lo))}")
        print("     -> IoU 會被高前景的 class 灌水（全predict前景就有高分），"
              "跨 class 的平均 IoU 不可直接比較")


if __name__ == "__main__":
    main()