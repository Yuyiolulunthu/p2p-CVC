"""
把 SEM 與 GT 並排畫出來看。

    python -m src.view_pairs --root dataset
    python -m src.view_pairs --root dataset --cls E2E_N --n 4

輸出 paper/pairs_<...>.png，每一列是一個樣本，四欄：

    SEM | GT | GT 輪廓疊在 SEM 上 | SEM 的灰階直方圖（依 GT 分成前景/背景）

第三欄是關鍵：如果 GT 真的是從這張 SEM 門檻化來的，紅色輪廓應該**貼合**
SEM 上的明暗交界。若輪廓整個偏移、或落在看不出對應關係的地方，就表示
GT 與 SEM 沒對齊、或根本不是從這張 SEM 來的。

第四欄回答另一個問題：前景與背景的灰階分布如果大幅重疊，任何門檻化都不可能成功。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import SEMDataset, scan_pairs

plt.rcParams.update({"font.size": 7, "figure.dpi": 130})


def contour(mask: np.ndarray) -> np.ndarray:
    """1 px 輪廓：自己與向右/向下位移一格比對。"""
    m = mask > 0.5
    e = np.zeros_like(m)
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:-1, :] |= m[:-1, :] != m[1:, :]
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--split", default="splits/split_seed1.json")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--cls", default=None, help="只看某個 class；預設各 class 取一張")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = scan_pairs(args.root)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    ds = SEMDataset(split[args.subset], items, augment=False)

    # 選樣本：指定 class 就取前 n 張，否則每個 class 各取一張
    picks, seen = [], set()
    for i in range(len(ds)):
        c = ds.items[i]["cls"]
        if args.cls:
            if c == args.cls and len(picks) < args.n:
                picks.append(i)
        elif c not in seen:
            seen.add(c)
            picks.append(i)
    picks = picks[:args.n] if args.cls else picks
    if not picks:
        raise SystemExit(f"找不到符合的樣本（cls={args.cls}）")

    fig, axes = plt.subplots(len(picks), 4,
                             figsize=(10, 2.3 * len(picks)))
    axes = np.atleast_2d(axes)

    for row, i in zip(axes, picks):
        b = ds[i]
        img, gt = b["raw"][0].numpy(), b["y"][0].numpy()
        g = gt > 0.5

        row[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[0].set_ylabel(f"{b['cls']}\n{Path(b['rel']).name}", fontsize=6)
        row[1].imshow(gt, cmap="gray", vmin=0, vmax=1)

        ov = np.stack([img] * 3, -1)
        ov[contour(gt)] = [1, 0, 0]
        row[2].imshow(ov)

        for ax in row[:3]:
            ax.set_xticks([]); ax.set_yticks([])

        # 灰階分布：GT 前景 vs 背景
        ax = row[3]
        bins = np.linspace(0, 1, 64)
        ax.hist(img[g], bins=bins, alpha=.6, label="GT fg", density=True)
        ax.hist(img[~g], bins=bins, alpha=.6, label="GT bg", density=True)
        d = float(img[g].mean() - img[~g].mean()) if g.any() and (~g).any() else 0
        ax.set_title(f"fg={g.mean():.2f}  d_gray={d:+.3f}", fontsize=6)
        ax.tick_params(labelsize=5)
        if i == picks[0]:
            ax.legend(fontsize=5)

    for ax, t in zip(axes[0], ["SEM", "GT", "GT contour on SEM", "gray histogram"]):
        ax.set_title(t, fontsize=8, pad=4)

    out = args.out or f"paper/pairs_{args.cls or 'allclass'}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved -> {out}")
    print("\n看第三欄：紅色輪廓有沒有貼合 SEM 上的明暗交界？")
    print("  貼合      -> GT 確實來自這張 SEM，問題在 cleaning 的強度")
    print("  整體偏移  -> 對齊問題")
    print("  完全無關  -> GT 不是從這張 SEM 產生的，要回去問學長")


if __name__ == "__main__":
    main()
