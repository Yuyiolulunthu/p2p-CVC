"""
把所有要用的 encoder 預訓練權重抓下來，存成一包可以整個搬到叢集的快取。

在**有網路的機器**上跑（你的筆電，或 Nano5 的登入節點）：

    python -m src.prefetch_weights --cache-dir ./weights_cache

然後把 weights_cache/ 一起傳上去，在 sbatch 裡設：

    export TORCH_HOME=$WORK/weights_cache/torch
    export HF_HOME=$WORK/weights_cache/hf
    export HF_HUB_OFFLINE=1          # 強制離線，有漏抓的會立刻報錯而不是卡住

為什麼需要這支：segmentation_models_pytorch 是在 build_model() 當下才去
HuggingFace 抓 ImageNet 權重的。計算節點沒有對外網路，整批 job 會在同一個
地方失敗，而且錯誤訊息是連線逾時，看起來像叢集壞掉。

跑完會做一次離線驗證（設 HF_HUB_OFFLINE=1 重新建一次模型），
確認快取真的完整 —— 這一步比下載本身重要。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 與 src/bench.py 的候選清單一致
ENCODERS = [
    "resnet34",
    "mobileone_s0",
    "mobileone_s1",
    "mit_b0",
    "efficientnet-b0",
    "timm-mobilenetv3_large_100",
]


def build_once(enc: str, in_ch: int) -> tuple[bool, str]:
    import segmentation_models_pytorch as smp
    try:
        smp.Unet(encoder_name=enc, encoder_weights="imagenet",
                 in_channels=in_ch, classes=1)
        return True, ""
    except Exception as e:                              # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="weights_cache")
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache_dir).resolve()
    (cache / "torch").mkdir(parents=True, exist_ok=True)
    (cache / "hf").mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["HF_HOME"] = str(cache / "hf")
    os.environ.pop("HF_HUB_OFFLINE", None)

    print(f"快取目錄: {cache}\n")
    ok, bad = [], []
    for enc in args.encoders:
        print(f"  抓取 {enc} ...", end=" ", flush=True)
        good, err = build_once(enc, args.in_channels)
        print("OK" if good else f"失敗 — {err}")
        (ok if good else bad).append(enc)

    size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    print(f"\n成功 {len(ok)} / {len(args.encoders)}，快取大小 {size/1e6:.1f} MB")
    if bad:
        print(f"失敗: {bad}")

    if args.no_verify or not ok:
        return

    # ---- 離線驗證：這一步才是重點 ----
    print("\n=== 離線驗證（HF_HUB_OFFLINE=1）===")
    code = (
        "import os,sys;"
        "import segmentation_models_pytorch as smp\n"
        "bad=[]\n"
        f"for e in {ok!r}:\n"
        "    try:\n"
        f"        smp.Unet(encoder_name=e, encoder_weights='imagenet',"
        f" in_channels={args.in_channels}, classes=1)\n"
        "        print('  OK  ', e)\n"
        "    except Exception as ex:\n"
        "        bad.append(e); print('  FAIL', e, type(ex).__name__)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-c", code], env=env)
    if r.returncode == 0:
        print("\n  -> 快取完整，可以搬上叢集")
    else:
        print("\n  ** 有 encoder 在離線模式下建不起來，搬上去會失敗。"
              "先在這裡解決，不要等到排隊之後才發現。")
        sys.exit(1)


if __name__ == "__main__":
    main()
