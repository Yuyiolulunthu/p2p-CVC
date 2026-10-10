from __future__ import annotations

import argparse
import csv
import statistics
import time

import torch

from .models import build_model, count_params, resolution_report

CANDIDATES = [
    ("resnet34", "imagenet"),                   # baseline / 對照組
    ("mit_b0", "imagenet"),
    ("mobileone_s1", "imagenet"),
    ("mobileone_s0", "imagenet"),
    ("efficientnet-b0", "imagenet"),
    ("timm-mobilenetv3_large_100", "imagenet"),
]


# --------------------------------------------------------------------------
def reparameterize(model) -> int:
    """
    把訓練時的多分支結構合併成推論形式（MobileOne / RepVGG 這類）。

    不做這一步就計時，量到的是一個永遠不會被部署的模型：MobileOne 訓練時
    每個 3x3 旁邊掛著 1x1 與 identity 分支，推論前要折進同一個 3x3。
    這正是 structural reparameterization 的全部賣點，漏掉等於替它報一個
    不公平的延遲。

    回傳被合併的模組數；0 表示這個 encoder 本來就沒有可合併的分支。
    """
    # timm 官方工具優先
    try:
        from timm.utils import reparameterize_model as _timm_rep
        before = sum(1 for m in model.modules()
                     if hasattr(m, "reparameterize") or hasattr(m, "fuse"))
        if before:
            rep = _timm_rep(model)
            return before if rep is not None else before
    except Exception:                                       # noqa: BLE001
        pass

    # 後備：自己走一遍，誰有 reparameterize()/fuse() 就呼叫誰
    n = 0
    for m in model.modules():
        fn = getattr(m, "reparameterize", None) or getattr(m, "fuse", None)
        if callable(fn):
            try:
                fn()
                n += 1
            except Exception:                               # noqa: BLE001
                pass
    return n


def measure_flops(model, size, in_ch, device):
    try:
        from thop import profile
    except ImportError:
        return None
    x = torch.randn(1, in_ch, size, size, device=device)
    try:
        macs, _ = profile(model, inputs=(x,), verbose=False)
    except Exception:                                       # noqa: BLE001
        return None
    return macs * 2 / 1e9


@torch.no_grad()
def measure_latency(model, size, in_ch, device, batch=1,
                    warmup=20, iters=100, amp=False):
    """
    回傳 (median_ms, p90_ms, peak_mem_GB)。

    報中位數而不是平均：GPU 上偶發的排程抖動會把平均往上拉，中位數穩定得多。
    p90 一起報，讓讀者看得出抖動幅度。
    """
    model.eval()
    x = torch.randn(batch, in_ch, size, size, device=device)
    ctx = (torch.autocast("cuda", dtype=torch.float16)
           if amp and device.startswith("cuda")
           else torch.autocast("cpu", enabled=False))

    with ctx:
        for _ in range(warmup):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(iters):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
            if device.startswith("cuda") else float("nan"))
    return statistics.median(times), times[int(0.9 * len(times)) - 1], peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512])
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--variant", default="v0")
    ap.add_argument("--amp", action="store_true", help="以 fp16 autocast 計時")
    ap.add_argument("--no-reparam", action="store_true",
                    help="不做 reparameterization（用來證明它的效果）")
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
            except Exception as e:                          # noqa: BLE001
                print(f"{enc:<30} BUILD FAIL: {e}")
        return

    try:
        import thop                                          # noqa: F401
    except ImportError:
        print("** thop 沒裝，GFLOPs 會是 n/a。pip install thop 之後重跑。\n")

    print(f"device={args.device}  batch={args.batch}  "
          f"dtype={'fp16(autocast)' if args.amp else 'fp32'}  "
          f"reparam={'off' if args.no_reparam else 'on'}\n")

    rows = []
    for size in args.sizes:
        for enc, w in CANDIDATES:
            try:
                model = build_model({"encoder": enc, "encoder_weights": w,
                                     "in_channels": args.in_ch,
                                     "variant": args.variant}).to(args.device)
            except Exception as e:                           # noqa: BLE001
                print(f"[SKIP] {enc}: {e}")
                continue

            # 參數量在合併前算（論文報的是模型大小，不是推論圖的大小）
            p_enc = round(count_params(model.net.encoder), 2)
            p_tot = round(count_params(model), 2)

            n_rep = 0 if args.no_reparam else reparameterize(model)

            g = measure_flops(model, size, args.in_ch, args.device)
            med, p90, peak = measure_latency(
                model, size, args.in_ch, args.device,
                batch=args.batch, amp=args.amp)

            row = {"encoder": enc, "input_size": size, "batch": args.batch,
                   "dtype": "fp16" if args.amp else "fp32",
                   "reparam_modules": n_rep,
                   "encoder_params_M": p_enc, "total_params_M": p_tot,
                   "GFLOPs": round(g, 2) if g else "n/a",
                   "latency_ms": round(med, 3),
                   "latency_p90_ms": round(p90, 3),
                   "img_per_s": round(args.batch / (med / 1000), 1),
                   "peak_mem_GB": round(peak, 3)}
            rows.append(row)
            print(row)
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("沒有任何 encoder 成功建起來")

    import pathlib
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nsaved -> {out}")

    rep = [r for r in rows if r["reparam_modules"] == 0
           and r["encoder"].startswith("mobileone")]
    if rep and not args.no_reparam:
        print("\n** 注意：MobileOne 的 reparam_modules = 0，表示合併沒有生效。"
           "\n   timm 版本可能不同，先確認 encoder 內有沒有 MobileOneBlock。")


if __name__ == "__main__":
    main()