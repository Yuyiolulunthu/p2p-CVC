

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import P2PDataset, load_pairs
from .metrics import compute_all


def otsu(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """全域單一門檻。作為「static thresholding」的代表。"""
    hist, edges = np.histogram(img, bins=bins, range=(0, 1))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(bins))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    t = edges[int(np.argmax(sigma_b))]
    return (img > t).astype(np.float32)


def _integral_stats(img, w):
    """以 integral image 求局部 mean / std,O(1) per pixel。"""
    pad = w // 2
    a = np.pad(img, pad, mode="reflect")
    ii = a.cumsum(0).cumsum(1)
    ii2 = (a ** 2).cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    ii2 = np.pad(ii2, ((1, 0), (1, 0)))
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    y0, y1, x0, x1 = y, y + w, x, x + w
    area = w * w
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    s2 = ii2[y1, x1] - ii2[y0, x1] - ii2[y1, x0] + ii2[y0, x0]
    mean = s / area
    var = np.maximum(s2 / area - mean ** 2, 0)
    return mean, np.sqrt(var)


def adaptive_mean(img, w=31, c=0.0):
    m, _ = _integral_stats(img, w)
    return (img > m - c).astype(np.float32)


def sauvola(img, w=31, k=0.2, R=0.5):
    """SEM 領域最常用的 local thresholding,是最強的古典對照組。"""
    m, s = _integral_stats(img, w)
    t = m * (1 + k * (s / R - 1))
    return (img > t).astype(np.float32)


def niblack(img, w=31, k=-0.2):
    m, s = _integral_stats(img, w)
    return (img > m + k * s).astype(np.float32)


METHODS = {"otsu": otsu, "adaptive_mean": adaptive_mean,
           "sauvola": sauvola, "niblack": niblack}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--split", default="splits/split_seed0.json")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--window", type=int, default=31)
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 張。先用 60 張確認量級,定稿時再跑完整 test set")
    ap.add_argument("--out", default="paper/classical.json")
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    pairs = load_pairs(args.pairs)
    ds = P2PDataset(split[args.subset], pairs, mode="full", augment=False)

    res = {}
    for name, fn in METHODS.items():
        acc, n = {}, 0
        N = min(args.limit, len(ds)) if args.limit else len(ds)
        for i in range(N):
            b = ds[i]
            img = b["raw"][0].numpy()
            pred = fn(img, args.window) if name != "otsu" else fn(img)
            p = torch.from_numpy(pred)[None, None]
            for k, v in compute_all(p, b["y"][None]).items():
                acc[k] = acc.get(k, 0.0) + float(v)
            n += 1
        res[name] = {k: v / n for k, v in acc.items()}
        print(f"{name:<14} IoU={res[name]['iou']:.4f}  BIoU={res[name]['biou_d2']:.4f}  "
              f"fp_blobs={res[name]['fp_blobs']:.1f}  ({n} imgs)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
