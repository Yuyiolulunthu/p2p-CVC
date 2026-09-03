

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import P2PDataset, load_pairs, lofo_split, make_split
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
    Transformer 系(mit_*, tinyvit)與 CNN 系用不同 lr —— 這是允許的 per-model 調整,
    但 **paper 的 Implementation Details 必須註明**,否則會被質疑比較不公平。
    """
    name = cfg["encoder"]
    is_tf = name.startswith("mit_") or "vit" in name
    lr = cfg.get("lr") or (6e-5 if is_tf else 1e-4)
    wd = cfg.get("weight_decay", 1e-4 if is_tf else 1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return opt, lr


@torch.no_grad()
def evaluate(model, loader, device, cfg, light=False) -> dict:
    """light=True 只算便宜指標(~18ms/張),訓練中每 epoch 用;
    完整版(~130ms/張)每 full_eval_every 個 epoch 與最終評估才跑。"""
    model.eval()
    acc, n = {}, 0
    for b in loader:
        x, raw, y = b["x"].to(device), b["raw"].to(device), b["y"].to(device)
        out = model(x, raw)
        pred = binarize(out["logits"])
        for i in range(pred.shape[0]):
            m = compute_all(pred[i:i + 1], y[i:i + 1],
                            border_band=cfg.get("border_band", 32),
                            min_blob_area=cfg.get("min_blob_area", 4),
                            light=light)
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default=None,
                    help="給定則跑 leave-one-family-out,該 pattern family 完全不參與訓練與 val")
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--set", nargs="*", default=[],
                    help="覆寫 config,格式 key=value")
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
        cfg["encoder"].replace("-", ""), cfg.get("variant", "v0"),
        f"lofo{args.family}" if args.family else None,
        f"tv{cfg.get('lambda_tv', 0)}" if cfg.get("lambda_tv") else None,
        f"s{args.seed}",
    ]))
    out_dir = Path(cfg.get("out_root", "runs")) / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    set_seed(args.seed)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ---- data ----
    pairs = load_pairs(cfg.get("pairs_file", args.pairs))
    if args.family:
        split = lofo_split(pairs, args.family, seed=args.seed)
        cfg["held_out_family"] = args.family
    else:
        sp = Path(cfg.get("split_file", "splits/split_seed0.json"))
        split = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else make_split(
            pairs, seed=0, out=sp)

    common = dict(pairs=pairs, crop_size=cfg.get("crop_size", 256))
    tr = P2PDataset(split["train"], mode="crop", augment=True,
                    crop_margin=cfg.get("crop_margin", 0), **common)
    va = P2PDataset(split["val"], mode="full", augment=False, **common)
    te = P2PDataset(split["test"], mode="full", augment=False, **common)
    print(f"data: train={len(tr)} val={len(va)} test={len(te)}")

    nw = cfg.get("workers", 4)

    def dl(ds, bs, sh):
        # persistent_workers：Windows 用 spawn 啟動 worker，每個 epoch 重啟
        # 要重新 import torch，開銷可觀。設 True 只付一次。
        return DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=nw,
                          pin_memory=torch.cuda.is_available(), drop_last=sh,
                          persistent_workers=nw > 0,
                          prefetch_factor=4 if nw > 0 else None)
    tl = dl(tr, cfg.get("batch_size", 8), True)
    vl = dl(va, 1, False)
    tel = dl(te, 1, False)

    # ---- model ----
    model = build_model(cfg).to(device)
    opt, lr = make_optimizer(model, cfg)
    epochs = cfg.get("epochs", 100)
    warm = cfg.get("warmup_epochs", 5 if cfg["encoder"].startswith("mit_") else 0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warm, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    print(f"[{exp}] encoder={cfg['encoder']} variant={cfg.get('variant','v0')} "
          f"lr={lr} params: enc={count_params(model.net.encoder):.2f}M "
          f"total={count_params(model):.2f}M")

    log_path = out_dir / "metrics.csv"
    writer, fcsv = None, open(log_path, "w", newline="", encoding="utf-8")
    best, best_ep, patience = -1.0, -1, cfg.get("patience", 20)

    for ep in range(epochs):
        model.train()
        if ep < warm:
            for g in opt.param_groups:
                g["lr"] = lr * (ep + 1) / max(warm, 1)
        t0 = time.time()
        run = 0.0
        for b in tl:
            x, raw, y = b["x"].to(device), b["raw"].to(device), b["y"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                out = model(x, raw)
                loss, parts = seg_loss(out, y,
                                       cfg.get("lambda_tv", 0.0),
                                       cfg.get("lambda_r", 0.0))
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
            # 第 0 epoch 是 full eval,欄位最齊,以它定義 CSV header
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

    # ---- 最終:載回 best,在 test 上跑一次(且只跑這一次)----
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    final = {"val": evaluate(model, vl, device, cfg),
             "test": evaluate(model, tel, device, cfg),
             "encoder_params_M": count_params(model.net.encoder),
             "total_params_M": count_params(model),
             "best_epoch": best_ep, "config": cfg}
    (out_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False))
    print(json.dumps(final["test"], indent=2))


if __name__ == "__main__":
    main()