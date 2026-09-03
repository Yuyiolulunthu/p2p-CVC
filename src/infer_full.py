

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import P2PDataset, load_pairs
from .metrics import binarize, compute_all
from .models import build_model


def _gauss_window(h, w, sigma_ratio=0.125, device="cpu"):
    """中央權重高、邊緣趨近 0,用來消除 tile 接縫。"""
    yy = torch.linspace(-1, 1, h, device=device)[:, None]
    xx = torch.linspace(-1, 1, w, device=device)[None, :]
    s = sigma_ratio * 8
    g = torch.exp(-(yy ** 2 + xx ** 2) / (2 * (s ** 2)))
    return (g / g.max())[None, None]


@torch.no_grad()
def predict(model, x, raw, mode="whole", tile=512, overlap=0.25, pad_to=32):
    """回傳 logits,尺寸與輸入相同。"""
    B, _, H, W = x.shape
    dev = x.device

    if mode == "whole":
        ph = (pad_to - H % pad_to) % pad_to
        pw = (pad_to - W % pad_to) % pad_to
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        rp = F.pad(raw, (0, pw, 0, ph), mode="reflect")
        return model(xp, rp)["logits"][..., :H, :W]

    step = tile if mode == "tile" else int(tile * (1 - overlap))
    acc = torch.zeros(B, 1, H, W, device=dev)
    wsum = torch.zeros(B, 1, H, W, device=dev)
    win = _gauss_window(tile, tile, device=dev) if mode == "blend" else 1.0

    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)

    for y in ys:
        for xx0 in xs:
            xc = x[..., y:y + tile, xx0:xx0 + tile]
            rc = raw[..., y:y + tile, xx0:xx0 + tile]
            lg = model(xc, rc)["logits"]
            acc[..., y:y + tile, xx0:xx0 + tile] += lg * win
            wsum[..., y:y + tile, xx0:xx0 + tile] += win
    return acc / wsum.clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", nargs="+", default=["whole", "tile", "blend"])
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="paper/infer_strategies.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    split = json.loads(Path(cfg.get("split_file", "splits/split_seed0.json")).read_text(encoding="utf-8"))
    pairs = load_pairs(cfg.get("pairs_file", "pairs.json"))
    ds = P2PDataset(split[args.split], pairs, mode="full", augment=False)

    results = {}
    for mode in args.mode:
        acc, n, t0 = {}, 0, time.perf_counter()
        for i in range(len(ds)):
            b = ds[i]
            x = b["x"][None].to(device)
            raw = b["raw"][None].to(device)
            y = b["y"][None].to(device)
            pred = binarize(predict(model, x, raw, mode, args.tile, args.overlap))
            for k, v in compute_all(pred, y).items():
                acc[k] = acc.get(k, 0.0) + float(v)
            n += 1
        r = {k: v / n for k, v in acc.items()}
        r["sec_per_image"] = (time.perf_counter() - t0) / n
        results[mode] = r
        print(mode, json.dumps({k: round(v, 4) for k, v in r.items()}, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
