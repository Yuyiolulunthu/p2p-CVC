"""
統一訓練入口。Phase 1 / 2 / 3 全部走這一支，只換 config。

    # baseline
    python -m src.train --config configs/base.yaml --seed 0 --set encoder=mobileone_s1 lr=0.0002

    # Phase 2：threshold head
    python -m src.train --config configs/base.yaml --seed 0 --set variant=v4 lambda_tv=0.01

    # Phase 3：leave-one-class-out
    python -m src.train --config configs/base.yaml --seed 0 --class E2E_N

每個 run 產出 runs/<name>/{config.yaml, metrics.csv, best.pt, final_metrics.json}
final_metrics.json 內含 per-class 明細 —— 這份資料的前景佔比從 0.10 到 0.89，
只看整體平均會被高前景的 class 灌水，所以 per-class 是必要的。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import SEMDataset, loco_split, make_split, scan_pairs
from .losses import seg_loss
from .metrics import binarize, compute_all
from .models import build_model, count_params


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # noqa: BLE001
        return "nogit"


def make_optimizer(model, cfg):
    """
    Transformer 系與 CNN 系的 lr 差一個量級。這是允許的 per-model 調整，
    但論文的 Implementation Details 必須註明。
    """
    name = cfg["encoder"]
    is_tf = name.startswith("mit_") or "vit" in name
    lr = cfg.get("lr") or (3e-4 if is_tf else 2e-4)
    wd = cfg.get("weight_decay", 1e-4)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd), lr


@torch.no_grad()
def evaluate(model, loader, device, cfg, light=False, per_class=False):
    """
    light=True 只算便宜的指標（每 epoch 用）；完整版含 EPE / CD / FP blob。
    per_class=True 時額外回傳每個 class 的明細。
    """
    model.eval()
    acc, n = defaultdict(float), 0
    pc_acc, pc_n = defaultdict(lambda: defaultdict(float)), defaultdict(int)

    for b in loader:
        x, raw, y = b["x"].to(device), b["raw"].to(device), b["y"].to(device)
        pred = binarize(model(x, raw)["logits"])
        for i in range(pred.shape[0]):
            m = compute_all(pred[i:i + 1], y[i:i + 1],
                            border_band=cfg.get("border_band", 32),
                            min_blob_area=cfg.get("min_blob_area", 4),
                            light=light)
            for k, v in m.items():
                acc[k] += float(v)
            n += 1
            if per_class:
                c = b["cls"][i]
                pc_n[c] += 1
                for k, v in m.items():
                    pc_acc[c][k] += float(v)

    out = {k: v / max(n, 1) for k, v in acc.items()}
    if per_class:
        out = {"overall": out, "per_class": {
            c: {**{k: v / pc_n[c] for k, v in d.items()}, "n": pc_n[c]}
            for c, d in pc_acc.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class", dest="held_out", default=None,
                    help="給定則跑 leave-one-class-out，該 class 完全不參與訓練與 val")
    ap.add_argument("--set", nargs="*", default=[], help="覆寫 config，key=value")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:                               # noqa: BLE001
            pass
        cfg[k] = v
    cfg["seed"] = args.seed
    cfg["git"] = git_hash()

    exp = args.name or "_".join(filter(None, [
        str(cfg["encoder"]).replace("-", ""), str(cfg.get("variant", "v0")),
        f"loco{args.held_out}" if args.held_out else None,
        f"tv{cfg.get('lambda_tv')}" if cfg.get("lambda_tv") else None,
        f"s{args.seed}",
    ]))
    out_dir = Path(cfg.get("out_root", "runs")) / exp
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    items = scan_pairs(cfg.get("data_root", "dataset"),
                       cfg.get("input_dir", "input"), cfg.get("gt_dir", "gt"))
    if args.held_out:
        split = loco_split(items, args.held_out, seed=args.seed)
        cfg["held_out_class"] = args.held_out
    else:
        sp = Path(cfg.get("split_file", "splits/split_seed1.json"))
        split = (json.loads(sp.read_text(encoding="utf-8")) if sp.exists()
                 else make_split(items, seed=cfg.get("split_seed", 1), out=sp))

    size = cfg.get("image_size", 512)
    cs = cfg.get("crop_size", 0)            # 0 = 整張訓練
    tr = SEMDataset(split["train"], items, size=size, augment=True,
                    crop_size=cs, crop_margin=cfg.get("crop_margin", 0))
    va = SEMDataset(split["val"], items, size=size, augment=False)
    te = SEMDataset(split["test"], items, size=size, augment=False)
    print(f"data: train={len(tr)} val={len(va)} test={len(te)}"
          + (f"  [held out: {args.held_out}]" if args.held_out else ""))

    nw = cfg.get("workers", 4)

    def dl(ds, bs, sh):
        # persistent_workers：Windows 用 spawn 啟動 worker，每個 epoch 重啟
        # 要重新 import torch，開銷可觀。設 True 只付一次。
        return DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=nw,
                          pin_memory=torch.cuda.is_available(), drop_last=sh,
                          persistent_workers=nw > 0,
                          prefetch_factor=4 if nw > 0 else None)

    tl, vl, tel = dl(tr, cfg.get("batch_size", 8), True), dl(va, 1, False), dl(te, 1, False)

    # ---- model ----
    model = build_model(cfg).to(device)
    opt, lr = make_optimizer(model, cfg)
    epochs = cfg.get("epochs", 100)
    warm = cfg.get("warmup_epochs", 5 if str(cfg["encoder"]).startswith("mit_") else 0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warm, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    cfg["lr_used"] = lr
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    print(f"[{exp}] encoder={cfg['encoder']} variant={cfg.get('variant','v0')} "
          f"lr={lr} params: enc={count_params(model.net.encoder):.2f}M "
          f"total={count_params(model):.2f}M")

    writer, fcsv = None, open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8")
    best, best_ep, patience = -1.0, -1, cfg.get("patience", 20)

    for ep in range(epochs):
        model.train()
        if ep < warm:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / max(warm, 1)
        t0, run = time.time(), 0.0

        for b in tl:
            x, raw, y = b["x"].to(device), b["raw"].to(device), b["y"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                out = model(x, raw)
                loss, parts = seg_loss(out, y,
                                       cfg.get("lambda_tv", 0.0),
                                       cfg.get("lambda_r", 0.0),
                                       cfg.get("w_bce", 1.0),
                                       cfg.get("w_dice", 0.5))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += parts["total"]
        if ep >= warm:
            sched.step()

        full = (ep % cfg.get("full_eval_every", 10) == 0) or (ep == epochs - 1)
        m = evaluate(model, vl, device, cfg, light=not full)
        row = {"epoch": ep, "train_loss": run / max(len(tl), 1),
               "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 1), **m}
        if writer is None:
            # 第 0 epoch 是 full eval，欄位最齊，以它定義 CSV header
            writer = csv.DictWriter(fcsv, fieldnames=list(row.keys()),
                                    extrasaction="ignore", restval="")
            writer.writeheader()
        writer.writerow(row)
        fcsv.flush()
        extra = (f" FPblob={m['fp_blobs']:.1f} EPE={m['epe_mean']:.2f}"
                 if "fp_blobs" in m else "")
        print(f"  ep{ep:3d} loss={row['train_loss']:.4f} IoU={m['iou']:.4f} "
              f"BIoU={m['biou_d2']:.4f}{extra}")

        if m["iou"] > best:
            best, best_ep = m["iou"], ep
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep},
                       out_dir / "best.pt")
        elif ep - best_ep >= patience:
            print(f"  early stop @ {ep} (best {best:.4f} @ {best_ep})")
            break
    fcsv.close()

    # ---- 載回 best，在 test 上跑一次（而且只跑這一次）----
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device,
                                     weights_only=False)["model"])
    te_res = evaluate(model, tel, device, cfg, per_class=True)
    final = {"val": evaluate(model, vl, device, cfg),
             "test": te_res["overall"], "test_per_class": te_res["per_class"],
             "encoder_params_M": count_params(model.net.encoder),
             "total_params_M": count_params(model),
             "best_epoch": best_ep, "config": cfg}
    (out_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== test (overall) ===")
    print(json.dumps({k: round(v, 4) for k, v in final["test"].items()}, indent=2))
    print("\n=== test (per class) ===")
    print(f"{'class':<16}{'n':>5}{'IoU':>9}{'B-IoU':>9}{'EPE':>8}")
    for c, d in sorted(final["test_per_class"].items(), key=lambda kv: -kv[1]["iou"]):
        print(f"{c:<16}{d['n']:>5}{d['iou']:>9.4f}{d['biou_d2']:>9.4f}{d['epe_mean']:>8.2f}")


if __name__ == "__main__":
    main()