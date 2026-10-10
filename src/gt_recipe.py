"""
反推 GT 的產生配方。

    python -m src.gt_recipe --root dataset --split splits/split_seed1.json

學長說 GT = thresholding + cleaning。這支腳本把這句話拆開量化：

  1. oracle 全域門檻  —— 每張圖掃過所有門檻值，取最好的那個 IoU。
     這是「任何全域門檻化方法的上限」。Otsu 只是其中一個選門檻的策略，
     oracle 告訴我們就算門檻選得完美能到多少。

  2. 先平滑再門檻   —— 量「雜訊」佔多少差距。

  3. 門檻後做形態學清理 —— 量「cleaning」佔多少差距。

判讀：
  oracle 就很高（>0.9）    -> GT 確實是門檻化來的，模型只要學會選門檻
  oracle 中等、清理後大增   -> cleaning 是主要工作，這才是模型的價值所在
  三個都不高               -> GT 不是從 SEM 門檻化來的，要回去問學長

只用 numpy + scipy，IoU 直接算，不依賴 src/metrics.py。
全域門檻用直方圖的累積和一次算完 256 個門檻，所以很快。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import SEMDataset, scan_pairs

BINS = 256


def iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0.5, gt > 0.5
    inter = np.count_nonzero(p & g)
    union = np.count_nonzero(p | g)
    return inter / max(union, 1)


def best_global_threshold(img: np.ndarray, gt: np.ndarray):
    """
    一次算出所有門檻的 IoU，回傳 (最佳 IoU, 最佳門檻)。

    pred(t) = img > t。令 h_all[k] 為灰階落在 bin k 的像素數，
    h_fg[k] 為其中 gt=1 的數量，則由高到低的累積和就是
    pred_sum(t) 與 inter(t)，IoU 可直接組出來。
    """
    q = np.clip((img * (BINS - 1)).astype(np.int32), 0, BINS - 1)
    g = gt > 0.5
    h_all = np.bincount(q.ravel(), minlength=BINS).astype(np.float64)
    h_fg = np.bincount(q[g].ravel(), minlength=BINS).astype(np.float64)

    pred_sum = np.cumsum(h_all[::-1])[::-1]     # pred_sum[k] = #(q >= k)
    inter = np.cumsum(h_fg[::-1])[::-1]
    gt_sum = float(g.sum())
    union = pred_sum + gt_sum - inter
    iou = inter / np.maximum(union, 1)
    k = int(np.argmax(iou))
    return float(iou[k]), k / (BINS - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--split", default="splits/split_seed1.json")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="paper/gt_recipe.json")
    args = ap.parse_args()

    try:
        from scipy import ndimage
    except ImportError:
        raise SystemExit("需要 scipy：pip install scipy")

    items = scan_pairs(args.root)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    ds = SEMDataset(split[args.subset], items, augment=False)
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    print(f"{args.subset}: {n} 張\n")

    rows = []
    for i in range(n):
        b = ds[i]
        img = b["raw"][0].numpy()
        gt = b["y"][0].numpy()
        r = {"cls": b["cls"]}

        # 1. oracle 全域門檻
        r["oracle"], r["t_opt"] = best_global_threshold(img, gt)

        # 2. 先高斯平滑再門檻（量雜訊的影響）
        for s in (1.0, 2.0):
            sm = ndimage.gaussian_filter(img, s)
            r[f"oracle_blur{s:g}"] = best_global_threshold(sm, gt)[0]

        # 3. 門檻後做形態學清理（量 cleaning 的影響）
        pred = (img > r["t_opt"]).astype(np.uint8)
        for k in (3, 5, 9):
            st = np.ones((k, k), bool)
            cleaned = ndimage.binary_closing(
                ndimage.binary_opening(pred, st), st)
            r[f"clean{k}"] = iou_np(cleaned, gt)

        # 4. 兩者都做
        sm = ndimage.gaussian_filter(img, 2.0)
        _, t2 = best_global_threshold(sm, gt)
        p2 = (sm > t2).astype(np.uint8)
        st = np.ones((5, 5), bool)
        r["blur2_clean5"] = iou_np(
            ndimage.binary_closing(ndimage.binary_opening(p2, st), st), gt)

        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")

    keys = ["oracle", "oracle_blur1", "oracle_blur2",
            "clean3", "clean5", "clean9", "blur2_clean5"]
    label = {"oracle": "oracle 全域門檻", "oracle_blur1": "  + 高斯平滑 σ=1",
             "oracle_blur2": "  + 高斯平滑 σ=2", "clean3": "  + 開閉運算 3x3",
             "clean5": "  + 開閉運算 5x5", "clean9": "  + 開閉運算 9x9",
             "blur2_clean5": "  平滑 σ=2 + 開閉 5x5"}

    print(f"\n=== 整體（{n} 張平均 IoU）===")
    base = float(np.mean([r["oracle"] for r in rows]))
    for k in keys:
        v = float(np.mean([r[k] for r in rows]))
        d = v - base
        print(f"{label[k]:<24}{v:>8.4f}" + (f"   ({d:+.4f})" if k != "oracle" else ""))

    t = np.array([r["t_opt"] for r in rows])
    print(f"\n最佳門檻值分布: median={np.median(t):.3f} "
          f"p10={np.percentile(t,10):.3f} p90={np.percentile(t,90):.3f}")
    print("  -> 若集中在一個窄範圍，表示有一個固定門檻能通用；"
          "分散則表示門檻必須隨影像調整")

    print(f"\n=== per-class（oracle）===")
    pc = defaultdict(list)
    for r in rows:
        pc[r["cls"]].append(r["oracle"])
    print(f"{'class':<16}{'n':>5}{'oracle IoU':>12}")
    for c, v in sorted(pc.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{c:<16}{len(v):>5}{np.mean(v):>12.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
         "t_opt_median": float(np.median(t)),
         "per_class_oracle": {c: float(np.mean(v)) for c, v in pc.items()}},
        indent=2), encoding="utf-8")
    print(f"\nsaved -> {args.out}")

    best = max(float(np.mean([r[k] for r in rows])) for k in keys)
    print(f"""
=== 判讀 ===
  最佳組合 IoU = {best:.3f}
  > 0.90  GT 就是門檻化（可能加簡單清理），模型主要學「選門檻」
  0.6-0.9 cleaning 是主要工作，這是模型真正的價值
  < 0.60  GT 不是從 SEM 門檻化來的，先回去問學長 cleaning 到底做了什麼""")


if __name__ == "__main__":
    main()
