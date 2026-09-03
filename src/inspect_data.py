

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def imgs(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def describe(p: Path) -> dict:
    im = Image.open(p)
    a = np.array(im)
    u = np.unique(a)
    return {
        "size": im.size,            # (W, H)
        "mode": im.mode,
        "dtype": str(a.dtype),
        "n_unique": int(u.size),
        "min": float(a.min()),
        "max": float(a.max()),
        # 這一項是關鍵：只有兩種值 -> binary mask；上百種 -> 灰階影像
        "looks_binary": bool(u.size <= 2),
        "fg_ratio": float((a > (a.max() / 2)).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/p2p/data")
    ap.add_argument("--n", type=int, default=3, help="每個資料夾取樣幾張")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"找不到 {root}")

    groups = sorted([d for d in root.iterdir() if d.is_dir()],
                    key=lambda d: (len(d.name), d.name))
    print(f"\n=== data root: {root} ===")
    print(f"group 資料夾共 {len(groups)} 個: {[g.name for g in groups]}")
    print("  -> 若是 12 個，這就是 Phase 3 LOCO 的 12 個 class\n")

    total = Counter()
    for g in groups:
        subs = sorted([d for d in g.iterdir() if d.is_dir()])
        print(f"--- {g.name} ---")
        if not subs:
            subs = [g]
        stems = {}
        for s in subs:
            fs = imgs(s)
            total[s.name] += len(fs)
            print(f"  {s.name:<34} {len(fs):>5} files")
            stems[s.name] = {p.stem for p in fs}
            for p in fs[:args.n]:
                d = describe(p)
                tag = "BINARY(=可能是 mask)" if d["looks_binary"] else "灰階(=可能是影像/DOM)"
                print(f"      {p.name:<40} {d['size']} {d['mode']} "
                      f"uniq={d['n_unique']:<5} rng=[{d['min']:.0f},{d['max']:.0f}] "
                      f"fg={d['fg_ratio']:.3f}  {tag}")

        # 檔名對應檢查
        if len(stems) == 2:
            a, b = list(stems)
            inter = stems[a] & stems[b]
            print(f"  檔名對應: {a} {len(stems[a])} / {b} {len(stems[b])} / 交集 {len(inter)}")
            if len(inter) < min(len(stems[a]), len(stems[b])):
                ex_a = sorted(stems[a] - stems[b])[:3]
                ex_b = sorted(stems[b] - stems[a])[:3]
                print(f"    ** 檔名不完全一致，需要寫對照規則。範例 {a}: {ex_a} / {b}: {ex_b}")
        print()

    print("=== 合計 ===")
    for k, v in total.items():
        print(f"  {k:<34} {v}")

    print("\n=== 判讀方式 ===")
    print("  pattern_image 若 looks_binary=True  -> 它是 GT mask，直接當 label")
    print("  pattern_image 若 looks_binary=False -> 它是 DOM reference（structural prior），")
    print("                                        那 GT mask 在別的地方，要先找出來")
    print("  兩邊 size 若不同 -> 需要 resize / 對齊，這會影響 IoU，要先處理")


if __name__ == "__main__":
    main()
