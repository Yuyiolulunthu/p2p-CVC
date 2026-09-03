
from __future__ import annotations

import argparse
import csv
import time

import torch

from .models import build_model, count_params, resolution_report

CANDIDATES = [
    ("resnet34", "imagenet"),                   # baseline / 對照組
    ("mit_b0", "imagenet"),                     # 候選 A ★ 主力
    ("mobileone_s1", "imagenet"),               # 備選:純 CNN 輕量對照
    ("mobileone_s0", "imagenet"),
    ("efficientnet-b0", "imagenet"),            # 備選:最常見 baseline
    ("timm-mobilenetv3_large_100", "imagenet"),
    # tinyvit_5m / ege_unet 不在 smp,見 src/models/tinyvit_encoder.py
]


def measure_flops(model, size, in_ch, device):
    try:
        from thop import profile
    except ImportError:
        return None
    x = torch.randn(1, in_ch, size, size, device=device)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return macs * 2 / 1e9


@torch.no_grad()
def measure_latency(model, size, in_ch, device, warmup=10, iters=50):
    model.eval()
    x = torch.randn(1, in_ch, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if device.startswith("cuda") else float("nan"))
    return ms, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512])
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--variant", default="v0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="paper/bench.csv")
    ap.add_argument("--check-resolution", action="store_true")
    args = ap.parse_args()

    if args.check_resolution:
        print(f"{'encoder':<30} " + " ".join(f"{s:>16}" for s in
                                             (256, 384, 512, 768, 1024)))
        for enc, w in CANDIDATES:
            try:
                m = build_model({"encoder": enc, "encoder_weights": w,
                                 "in_channels": args.in_ch, "variant": args.variant})
                r = resolution_report(m, in_ch=args.in_ch, device="cpu")
                print(f"{enc:<30} " + " ".join(f"{r[s]:>16}" for s in r))
            except Exception as e:                      # noqa: BLE001
                print(f"{enc:<30} BUILD FAIL: {e}")
        return

    rows = []
    for size in args.sizes:
        for enc, w in CANDIDATES:
            try:
                model = build_model({"encoder": enc, "encoder_weights": w,
                                     "in_channels": args.in_ch,
                                     "variant": args.variant}).to(args.device)
            except Exception as e:                      # noqa: BLE001
                print(f"[SKIP] {enc}: {e}")
                continue
            g = measure_flops(model, size, args.in_ch, args.device)
            ms, peak = measure_latency(model, size, args.in_ch, args.device)
            row = {"encoder": enc, "input_size": size,
                   "encoder_params_M": round(count_params(model.net.encoder), 2),
                   "total_params_M": round(count_params(model), 2),
                   "GFLOPs": round(g, 2) if g else "n/a",
                   "latency_ms": round(ms, 2), "peak_mem_GB": round(peak, 2)}
            rows.append(row)
            print(row)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
