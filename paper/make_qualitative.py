

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import P2PDataset, load_pairs
from src.metrics import binarize, cd_error, compute_all
from src.models import build_model

plt.rcParams.update({"font.family": "serif", "font.size": 7, "figure.dpi": 300})


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def predict(model, x, raw):
    return binarize(model(x, raw)["logits"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="LLWS")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--indist", default="runs/mobileone_s1_s0/best.pt",
                    help="見過所有 family 的模型")
    ap.add_argument("--lofo", default=None,
                    help="預設 runs/lofo_<FAMILY>/best.pt")
    ap.add_argument("--out", default="paper/fig3_failure.pdf")
    args = ap.parse_args()

    lofo_ckpt = args.lofo or f"runs/lofo_{args.family}/best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = load_pairs(args.pairs)
    stems = [p["stem"] for p in pairs if p["family"] == args.family]
    if not stems:
        raise SystemExit(f"family {args.family} 沒有影像")
    ds = P2PDataset(stems, pairs, mode="full", augment=False)
    print(f"{args.family}: {len(ds)} images")

    m_in = load(args.indist, device)
    m_lo = load(lofo_ckpt, device)

    # ---- 先算全部，依 LOFO 模型的 CD bias 絕對值排序 ----
    rows = []
    for i in range(len(ds)):
        b = ds[i]
        x = b["x"][None].to(device)
        raw = b["raw"][None].to(device)
        y = b["y"][None].to(device)
        p_in = predict(m_in, x, raw)
        p_lo = predict(m_lo, x, raw)
        rows.append({
            "i": i, "stem": b["stem"],
            "bias_lo": cd_error(p_lo, y)["cd_bias"],
            "iou_in": compute_all(p_in, y, light=True)["iou"],
            "iou_lo": compute_all(p_lo, y, light=True)["iou"],
            "img": b["raw"][0].numpy(), "gt": b["y"][0].numpy(),
            "p_in": p_in[0, 0].cpu().numpy(), "p_lo": p_lo[0, 0].cpu().numpy(),
        })
    rows.sort(key=lambda r: -abs(r["bias_lo"]))
    sel = rows[:args.n]
    print(f"selected: {[(r['stem'], round(r['bias_lo'],1)) for r in sel]}")

    # ---- 繪圖 ----
    cols = ["CD-SEM input", "Ground truth", "Trained on all families",
            f"{args.family} held out", "Error (LOFO)"]
    fig, axes = plt.subplots(len(sel), 5, figsize=(7.0, 1.45 * len(sel)))
    if len(sel) == 1:
        axes = axes[None]

    for r, axrow in zip(sel, axes):
        # error map：紅=假陽性(過胖)，藍=假陰性(過瘦)
        err = np.zeros((*r["gt"].shape, 3), dtype=np.float32)
        err[..., 0] = ((r["p_lo"] > 0.5) & (r["gt"] < 0.5)).astype(np.float32)
        err[..., 2] = ((r["p_lo"] < 0.5) & (r["gt"] > 0.5)).astype(np.float32)
        err += 0.12 * np.stack([r["img"]] * 3, -1) * (err.sum(-1, keepdims=True) == 0)

        panels = [(r["img"], "gray"), (r["gt"], "gray"),
                  (r["p_in"], "gray"), (r["p_lo"], "gray"), (err, None)]
        for ax, (im, cm) in zip(axrow, panels):
            ax.imshow(im, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
        axrow[0].set_ylabel(r["stem"], fontsize=5.5)
        axrow[2].set_xlabel(f"IoU {r['iou_in']:.3f}", fontsize=5.5, labelpad=1.5)
        axrow[3].set_xlabel(f"IoU {r['iou_lo']:.3f}   CD bias {r['bias_lo']:+.1f} px",
                            fontsize=5.5, labelpad=1.5)

    for ax, c in zip(axes[0], cols):
        ax.set_title(c, fontsize=6.5, pad=3)

    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.25)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)

    summary = {"family": args.family,
               "selected": [{k: r[k] for k in ("stem", "bias_lo", "iou_in", "iou_lo")}
                            for r in sel],
               "family_mean_bias": float(np.mean([r["bias_lo"] for r in rows]))}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
