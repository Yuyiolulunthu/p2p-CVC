"""
古典 thresholding 基準 + 極性驗證。

    python -m src.classical --root dataset --split splits/split_seed1.json

這支現在有兩個用途，第二個比第一個重要：

1. Table 5 的對照組（Otsu / Sauvola / Niblack / adaptive mean）

2. **驗證我們對 GT 的理解是否正確。** 學長說 GT 是用 thresholding + cleaning
   從 SEM 產生的。如果真是如此，古典 thresholding 在這份 GT 上應該拿到相當高的
   分數（遠高於先前對 DOM 的 0.44）。若仍然很低，表示我們對 GT 的理解還是錯的，
   必須先停下來弄清楚，不要又跑一輪白工。

同時檢查極性：GT 的白色對應 SEM 的亮區還是暗區。上一輪就是因為沒有先確認
目標是什麼而整批作廢，這次每個 class 都單獨驗。

純 numpy，不需要 scipy 以外的東西。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import SEMDataset, scan_pairs
from .metrics import compute_all


# ----------------------------------------------------------------------
# 古典方法
# ----------------------------------------------------------------------
def otsu(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """全域單一門檻，作為 static thresholding 的代表。"""
    hist, edges = np.histogram(img, bins=bins, range=(0, 1))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(bins))
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    return (img > edges[int(np.argmax(sigma_b))]).astype(np.float32)


def _integral_stats(img, w):
    """以 integral image 求局部 mean / std，每個像素 O(1)。"""
    pad = w // 2
    a = np.pad(img, pad, mode="reflect")
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ii2 = np.pad((a ** 2).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    y0, y1, x0, x1 = y, y + w, x, x + w
    area = w * w
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    s2 = ii2[y1, x1] - ii2[y0, x1] - ii2[y1, x0] + ii2[y0, x0]
    mean = s / area
    return mean, np.sqrt(np.maximum(s2 / area - mean ** 2, 0))


def adaptive_mean(img, w=31, c=0.0):
    m, _ = _integral_stats(img, w)
    return (img > m - c).astype(np.float32)


def sauvola(img, w=31, k=0.2, R=0.5):
    """SEM 領域最常用的 local thresholding，最強的古典對照組。"""
    m, s = _integral_stats(img, w)
    return (img > m * (1 + k * (s / R - 1))).astype(np.float32)


def niblack(img, w=31, k=-0.2):
    m, s = _integral_stats(img, w)
    return (img > m + k * s).astype(np.float32)


METHODS = {"otsu": lambda a, w: otsu(a),
           "adaptive_mean": adaptive_mean,
           "sauvola": sauvola,
           "niblack": niblack}


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--split", default="splits/split_seed1.json")
    ap.add_argument("--subset", default="test", choices=["train", "val", "test"])
    ap.add_argument("--window", type=int, default=31)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張（除錯用）")
    ap.add_argument("--out", default="paper/classical.json")
    args = ap.parse_args()

    items = scan_pairs(args.root)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    ds = SEMDataset(split[args.subset], items, augment=False)
    n_total = min(args.limit, len(ds)) if args.limit else len(ds)
    print(f"{args.subset}: {n_total} 張\n")

    # ---------- 1. 極性驗證（每個 class 分開）----------
    pol = defaultdict(list)
    for i in range(n_total):
        b = ds[i]
        img, gt = b["raw"][0].numpy(), b["y"][0].numpy() > 0.5
        if gt.all() or not gt.any():
            continue
        pol[b["cls"]].append(float(img[gt].mean() - img[~gt].mean()))

    print("=== 極性驗證：GT 白區在 SEM 上的平均亮度 − 黑區 ===")
    print(f"{'class':<16}{'n':>5}{'亮度差':>10}   判讀")
    bad = []
    for c, v in sorted(pol.items(), key=lambda kv: -np.mean(kv[1])):
        d = float(np.mean(v))
        same = sum(1 for x in v if x > 0)
        tag = "GT白 = SEM亮" if d > 0 else "** GT白 = SEM暗（極性相反）**"
        if d <= 0:
            bad.append(c)
        print(f"{c:<16}{len(v):>5}{d:>+10.4f}   {tag}  ({same}/{len(v)} 同號)")
    if bad:
        print(f"\n** 這些 class 極性與其他相反：{bad}")
        print("   古典方法的分數對它們沒有意義（會接近 1-IoU），先弄清楚再往下。")
    else:
        print("\n  -> 全部一致，古典方法可以直接套用")

    # ---------- 2. 古典方法 ----------
    print(f"\n=== 古典 thresholding（window={args.window}）===")
    res, per_cls = {}, {}
    for name, fn in METHODS.items():
        acc, n = defaultdict(float), 0
        pc = defaultdict(lambda: defaultdict(float))
        pc_n = defaultdict(int)
        for i in range(n_total):
            b = ds[i]
            pred = torch.from_numpy(fn(b["raw"][0].numpy(), args.window))[None, None]
            m = compute_all(pred, b["y"][None])
            for k, v in m.items():
                acc[k] += float(v)
                pc[b["cls"]][k] += float(v)
            pc_n[b["cls"]] += 1
            n += 1
        res[name] = {k: v / n for k, v in acc.items()}
        per_cls[name] = {c: {**{k: v / pc_n[c] for k, v in d.items()}, "n": pc_n[c]}
                         for c, d in pc.items()}
        r = res[name]
        print(f"{name:<14} IoU={r['iou']:.4f}  BIoU={r['biou_d2']:.4f}  "
              f"EPE={r['epe_mean']:.2f}  FPblob={r['fp_blobs']:.0f}")

    best = max(res, key=lambda k: res[k]["iou"])
    print(f"\n=== {best} 的 per-class 明細 ===")
    print(f"{'class':<16}{'n':>5}{'IoU':>9}{'B-IoU':>9}")
    for c, d in sorted(per_cls[best].items(), key=lambda kv: -kv[1]["iou"]):
        print(f"{c:<16}{d['n']:>5}{d['iou']:>9.4f}{d['biou_d2']:>9.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"overall": res, "per_class": per_cls,
         "polarity": {c: float(np.mean(v)) for c, v in pol.items()}},
        indent=2), encoding="utf-8")
    print(f"\nsaved -> {args.out}")

    print(f"""
=== 怎麼判讀 ===
  最佳古典方法 IoU = {res[best]['iou']:.3f}
  學長說 GT 是用 thresholding + cleaning 從 SEM 產生的。
  若此數字明顯高於 0.7，表示我們對 GT 的理解正確，可以往下做。
  若仍在 0.5 以下，表示 GT 不是單純由 SEM 門檻化而來，要先問清楚再跑訓練。""")


if __name__ == "__main__":
    main()