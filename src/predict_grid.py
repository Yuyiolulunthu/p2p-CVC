"""
用訓練好的 checkpoint 產出預測影像，分 class 存成資料夾。

    python -m src.predict_grid --run runs/timmmobilenetv3_large_100_s0
    python -m src.predict_grid --run runs/... --cls E2E_N --n 8
    python -m src.predict_grid --run runs/... --all          # 每一張都存

每張圖四欄：

    SEM | GT | 預測 | 誤差疊圖（紅=假陽性 FP，藍=假陰性 FN，綠=GT 輪廓）

預設每個 class 只挑三張：BIoU 最好 / 中位數 / 最差。看「最差那張」比看平均數字
更快知道模型在哪裡壞掉，而挑最好那張可以避免只看失敗案例產生的偏誤。

輸出：
    preds/<run 名稱>/<class>/best_<BIoU>_<檔名>.png      (單張四欄圖)
    preds/<run 名稱>/_summary_<class>.png                (該 class 三張併成一張)
    preds/<run 名稱>/per_image_biou.csv                  (每張圖的指標)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .data import SEMDataset, scan_pairs
from .metrics import compute_all
from .models import build_model

plt.rcParams.update({"font.size": 7, "figure.dpi": 150})


# ---------------------------------------------------------------- 載入
def load_cfg(run: Path) -> dict:
    fm = run / "final_metrics.json"
    if fm.exists():
        d = json.loads(fm.read_text(encoding="utf-8"))
        if isinstance(d.get("config"), dict):
            return d["config"]
    cy = run / "config.yaml"
    if cy.exists():
        return yaml.safe_load(cy.read_text(encoding="utf-8"))
    raise SystemExit(f"{run} 裡找不到 config（final_metrics.json 或 config.yaml）")


def load_weights(model, ckpt_path: Path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
            break
    missing, unexpected = model.load_state_dict(obj, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)} "
              f"— 權重與模型結構不完全吻合，確認 config 的 encoder/variant 正確")
    return model



def to_prob(out) -> torch.Tensor:
    """
    模型可能回傳 tensor 或 dict（v0 / v4 共用的包裝）。
    key 名稱帶 logit 的要過 sigmoid，帶 prob / mask 的已經是機率。
    """
    if torch.is_tensor(out):
        return torch.sigmoid(out)
    if isinstance(out, dict):
        for k in ("logit", "logits", "out", "y", "pred"):
            if k in out and torch.is_tensor(out[k]):
                return torch.sigmoid(out[k])
        for k in ("prob", "probs", "mask", "sigmoid"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        for k, v in out.items():
            if torch.is_tensor(v) and v.dim() == 4:
                print(f"  [info] 用 dict['{k}'] 當輸出（過 sigmoid）")
                return torch.sigmoid(v)
        raise SystemExit(f"模型輸出的 dict 裡找不到 4D tensor：{list(out)}")
    if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
        return torch.sigmoid(out[0])
    raise SystemExit(f"看不懂的模型輸出型別：{type(out)}")


# ---------------------------------------------------------------- 繪圖
def contour(mask: np.ndarray) -> np.ndarray:
    m = mask > 0.5
    e = np.zeros_like(m)
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:-1, :] |= m[:-1, :] != m[1:, :]
    return e


def panel(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7.5, pad=3)


def error_overlay(sem: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """SEM 上疊：紅 = 多畫的(FP)，藍 = 漏掉的(FN)，綠 = GT 輪廓。"""
    base = np.stack([sem * 0.75] * 3, -1)
    g, p = gt > 0.5, pred > 0.5
    base[p & ~g] = [0.92, 0.17, 0.17]
    base[~p & g] = [0.16, 0.38, 0.92]
    base[contour(gt)] = [0.10, 0.70, 0.35]
    return np.clip(base, 0, 1)


def draw_one(sem, gt, pred, meta, out: Path):
    fig, ax = plt.subplots(1, 4, figsize=(9.2, 2.55))
    panel(ax[0], sem, "SEM")
    panel(ax[1], gt, "Ground truth")
    panel(ax[2], pred, "Prediction")
    ax[3].imshow(error_overlay(sem, gt, pred), interpolation="nearest")
    ax[3].set_xticks([]); ax[3].set_yticks([])
    ax[3].set_title("Error  (red = FP, blue = FN, green = GT edge)", fontsize=7.5, pad=3)
    fig.suptitle(meta, fontsize=8, y=1.02)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def draw_summary(rows, cls, out: Path):
    """rows = [(tag, sem, gt, pred, meta), ...]"""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(9.2, 2.55 * n))
    axes = np.atleast_2d(axes)
    for r, (tag, sem, gt, pred, meta) in zip(axes, rows):
        panel(r[0], sem)
        panel(r[1], gt)
        panel(r[2], pred)
        r[3].imshow(error_overlay(sem, gt, pred), interpolation="nearest")
        r[3].set_xticks([]); r[3].set_yticks([])
        r[0].set_ylabel(f"{tag}\n{meta}", fontsize=6.5)
    for a, t in zip(axes[0], ["SEM", "Ground truth", "Prediction",
                              "Error (red FP / blue FN / green GT edge)"]):
        a.set_title(t, fontsize=8, pad=4)
    fig.suptitle(f"class: {cls}", fontsize=9.5, y=1.005)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name>")
    ap.add_argument("--root", default=None, help="覆寫 data_root")
    ap.add_argument("--split", default=None, help="覆寫 split 檔")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--cls", default=None, help="只做這個 class")
    ap.add_argument("--n", type=int, default=3, help="每個 class 存幾張（best/median/worst）")
    ap.add_argument("--all", action="store_true", help="每一張都存")
    ap.add_argument("--out", default="preds")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_cfg(run)
    root = args.root or cfg.get("data_root", "dataset")
    split_file = args.split or cfg.get("split_file", "splits/split_seed1.json")

    print(f"run      : {run.name}")
    print(f"encoder  : {cfg.get('encoder')}  variant={cfg.get('variant', 'v0')}")
    print(f"data     : {root}   split={split_file}")

    items = scan_pairs(root, verbose=False)
    held = cfg.get("held_out") or cfg.get("class") or cfg.get("held_out_class")
    if held:
        rels = [i["rel"] for i in items if i["cls"] == held]
        print(f"LOCO run — 測試集是留出的 class「{held}」，共 {len(rels)} 張")
    else:
        rels = json.loads(Path(split_file).read_text(encoding="utf-8"))[args.subset]

    ds = SEMDataset(rels, items, augment=False)

    ckpt = run / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"找不到 {ckpt}")
    model = build_model(cfg).to(args.device)
    load_weights(model, ckpt, args.device)
    model.eval()

    outdir = Path(args.out) / run.name
    outdir.mkdir(parents=True, exist_ok=True)

    per_cls: dict[str, list] = {}
    rows_csv = []

    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            cls = b["cls"]
            if args.cls and cls != args.cls:
                continue
            x = b["x"].unsqueeze(0).to(args.device)
            y = b["y"].unsqueeze(0).to(args.device)
            prob = to_prob(model(x))
            m = compute_all(prob, y)

            sem = b["raw"][0].numpy()
            gt = b["y"][0].numpy()
            pred = (prob[0, 0].cpu().numpy() > 0.5).astype(np.float32)

            meta = (f"{b['rel']}   IoU {m['iou']:.4f}   BIoU {m['biou_d2']:.4f}   "
                    f"EPE {m['epe_mean']:.2f}px   CD bias {m['cd_bias']:+.2f}px   "
                    f"FP blob {m['fp_blobs']:.0f}")
            per_cls.setdefault(cls, []).append((m["biou_d2"], sem, gt, pred, meta, b["rel"]))
            rows_csv.append({"rel": b["rel"], "cls": cls,
                             **{k: round(float(v), 5) for k, v in m.items()}})

            if args.all:
                draw_one(sem, gt, pred, meta,
                         outdir / cls / f"{Path(b['rel']).stem}.png")

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(ds)}")

    if not per_cls:
        raise SystemExit(f"沒有符合的樣本（--cls {args.cls}）")

    # 每個 class 挑 best / median / worst
    for cls, lst in sorted(per_cls.items()):
        lst.sort(key=lambda r: r[0])
        if len(lst) == 1:
            picks = [("only", lst[0])]
        elif args.n >= 3 and len(lst) >= 3:
            picks = [("worst", lst[0]),
                     ("median", lst[len(lst) // 2]),
                     ("best", lst[-1])]
        else:
            picks = [("worst", lst[0]), ("best", lst[-1])]

        rows = [(tag, r[1], r[2], r[3], f"BIoU {r[0]:.4f}\n{Path(r[5]).name}")
                for tag, r in picks]
        draw_summary(rows, cls, outdir / f"_summary_{cls}.png")
        for tag, r in picks:
            draw_one(r[1], r[2], r[3], f"[{tag}] {r[4]}",
                     outdir / cls / f"{tag}_biou{r[0]:.4f}_{Path(r[5]).stem}.png")
        print(f"  {cls:<14} n={len(lst):<4} "
              f"BIoU {lst[0][0]:.4f} … {lst[-1][0]:.4f}")

    with (outdir / "per_image_biou.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader(); w.writerows(rows_csv)

    print(f"\n-> {outdir}")
    print("   _summary_<class>.png 是每個 class 的 worst / median / best 三列併圖，"
          "最適合放進報告。")


if __name__ == "__main__":
    main()