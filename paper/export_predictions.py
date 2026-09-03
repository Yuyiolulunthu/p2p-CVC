import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import P2PDataset, load_pairs          # noqa: E402
from src.metrics import binarize, compute_all, erode  # noqa: E402
from src.models import build_model                   # noqa: E402

KINDS = ("mask", "overlay", "error", "panel")


def save(arr, path: Path):
    """arr 為 float [0,1] 灰階或 (H,W,3)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(arr, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def contour(mask: np.ndarray) -> np.ndarray:
    """取 1 px 輪廓，用來疊在原圖上。"""
    t = torch.from_numpy(mask)[None, None].float()
    return (t - erode(t, 3))[0, 0].numpy() > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--split", default="splits/split_seed0.json")
    ap.add_argument("--subset", default="all",
                    choices=["all", "train", "val", "test"])
    ap.add_argument("--family", default=None)
    ap.add_argument("--out-root", default="predictions")
    ap.add_argument("--name", default=None, help="輸出子資料夾名，預設用 run 名稱")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--good", type=float, default=0.90)
    ap.add_argument("--fair", type=float, default=0.80)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 張（除錯用）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = load_pairs(args.pairs)

    if args.subset == "all":
        stems = [p["stem"] for p in pairs]
    else:
        import json
        stems = json.loads(Path(args.split).read_text(encoding="utf-8"))[args.subset]
    if args.family:
        fam = {p["stem"] for p in pairs if p["family"] == args.family}
        stems = [s for s in stems if s in fam]
    ds = P2PDataset(stems, pairs, mode="full", augment=False)
    n_total = min(args.limit, len(ds)) if args.limit else len(ds)
    print(f"處理 {n_total} 張，裝置 {device}")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])

    root = Path(args.out_root) / (args.name or Path(args.ckpt).parent.name)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    by_fam = defaultdict(list)
    with torch.no_grad():
        for i in range(n_total):
            b = ds[i]
            x = b["x"][None].to(device)
            raw = b["raw"][None].to(device)
            y = b["y"][None].to(device)
            pred = binarize(model(x, raw)["logits"])
            m = compute_all(pred, y)

            stem, family = b["stem"], b["family"]
            img = b["raw"][0].numpy()
            gt = b["y"][0].numpy()
            pr = pred[0, 0].cpu().numpy()

            if "mask" in args.kinds:
                save(pr, root / "mask" / family / f"{stem}.png")

            if "overlay" in args.kinds:
                ov = np.stack([img] * 3, -1)
                cg, cp = contour(gt), contour(pr)
                ov[cg] = [1.0, 0.85, 0.1]      # GT 輪廓：黃
                ov[cp] = [0.1, 1.0, 0.3]       # 預測輪廓：綠
                save(ov, root / "overlay" / family / f"{stem}.png")

            err = np.zeros((*gt.shape, 3), np.float32)
            err[..., 0] = (pr > 0.5) & (gt < 0.5)
            err[..., 2] = (pr < 0.5) & (gt > 0.5)
            err += 0.15 * np.stack([img] * 3, -1) * (err.sum(-1, keepdims=True) == 0)
            if "error" in args.kinds:
                save(err, root / "error" / family / f"{stem}.png")

            if "panel" in args.kinds:
                g = 4                                   # 分隔線寬
                h, w = img.shape
                canvas = np.ones((h, w * 4 + g * 3, 3), np.float32) * 0.5
                for k, tile in enumerate([np.stack([img] * 3, -1),
                                          np.stack([gt] * 3, -1),
                                          np.stack([pr] * 3, -1), err]):
                    x0 = k * (w + g)
                    canvas[:, x0:x0 + w] = tile
                save(canvas, root / "panel" / family / f"{stem}.png")

            row = {"stem": stem, "family": family,
                   **{k: round(float(v), 5) for k, v in m.items()}}
            rows.append(row)
            by_fam[family].append(m["iou"])

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_total}")

    # ---- metrics.csv ----
    with open(root / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    # ---- 依品質分級複製 panel，方便快速瀏覽 ----
    if "panel" in args.kinds:
        for r in rows:
            iou = r["iou"]
            grade = "good" if iou >= args.good else ("fair" if iou >= args.fair else "poor")
            src = root / "panel" / r["family"] / f"{r['stem']}.png"
            dst = root / "by_quality" / grade / f"{iou:.3f}_{r['family']}_{r['stem']}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copyfile(src, dst)

    # ---- summary.txt ----
    lines = [f"checkpoint : {args.ckpt}",
             f"subset     : {args.subset}" + (f" / family={args.family}" if args.family else ""),
             f"images     : {len(rows)}", ""]
    ious = np.array([r["iou"] for r in rows])
    bious = np.array([r["biou_d2"] for r in rows])
    lines.append(f"overall    IoU {ious.mean():.4f}   B-IoU {bious.mean():.4f}")
    grades = {"good": (ious >= args.good).sum(),
              "fair": ((ious >= args.fair) & (ious < args.good)).sum(),
              "poor": (ious < args.fair).sum()}
    lines.append(f"grades     good {grades['good']}  fair {grades['fair']}  poor {grades['poor']}")
    lines.append("")
    lines.append(f"{'family':<10}{'n':>5}{'IoU':>9}{'B-IoU':>9}")
    for fam in sorted(by_fam, key=lambda k: -np.mean(by_fam[k])):
        sub = [r for r in rows if r["family"] == fam]
        lines.append(f"{fam:<10}{len(sub):>5}"
                     f"{np.mean([r['iou'] for r in sub]):>9.4f}"
                     f"{np.mean([r['biou_d2'] for r in sub]):>9.4f}")
    lines.append("")
    lines.append("最差 10 張：")
    for r in sorted(rows, key=lambda r: r["iou"])[:10]:
        lines.append(f"  {r['iou']:.4f}  {r['family']:<8}{r['stem']}")
    txt = "\n".join(lines)
    (root / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\n輸出 -> {root.resolve()}")


if __name__ == "__main__":
    main()
