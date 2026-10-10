"""

量出古典 thresholding 各步驟的實際耗時，找出瓶頸在哪。
"""

import json
import time

import torch

from src.classical import adaptive_mean, niblack, otsu, sauvola
from src.data import P2PDataset, load_pairs
from src.metrics import compute_all, fp_blob_count

# 先確認 scipy 分支有沒有被載到 —— 這是最常見的問題（舊檔沒覆蓋掉）
import src.metrics as M
print("scipy 分支存在:", hasattr(M, "_fp_blobs_scipy"))
try:
    import scipy
    print("scipy 版本:", scipy.__version__)
except ImportError:
    print("!! scipy 沒裝，fp_blobs 會慢 15 倍而且數值是錯的")
print("CUDA:", torch.cuda.is_available())
print()

pairs = load_pairs("pairs.json")
split = json.load(open("splits/split_seed0.json"))
ds = P2PDataset(split["test"], pairs, mode="full", augment=False)
b = ds[0]
img = b["raw"][0].numpy()
g = b["y"][None]
print(f"樣本 {b['stem']}  family={b['family']}  shape={img.shape}\n")

for name, fn in [("otsu", lambda: otsu(img)),
                 ("adaptive_mean", lambda: adaptive_mean(img, 31)),
                 ("sauvola", lambda: sauvola(img, 31)),
                 ("niblack", lambda: niblack(img, 31))]:
    t = time.perf_counter()
    p = fn()
    print(f"{name:<14} threshold {(time.perf_counter()-t)*1000:7.1f} ms  fg={p.mean():.3f}")

p = torch.from_numpy(otsu(img))[None, None]

t = time.perf_counter()
r = fp_blob_count(p, g)
print(f"\nfp_blob_count  {(time.perf_counter()-t)*1000:7.1f} ms  -> {r}")

t = time.perf_counter()
compute_all(p, g, light=True)
print(f"compute_all(light)  {(time.perf_counter()-t)*1000:7.1f} ms")

t = time.perf_counter()
m = compute_all(p, g)
dt = time.perf_counter() - t
print(f"compute_all(full)   {dt*1000:7.1f} ms")
print(f"\n=> 160 張 x 4 方法 ≈ {160*4*dt/60:.1f} 分鐘")
print(f"\n這張圖的 otsu 結果: IoU={m['iou']:.3f} BIoU={m['biou_d2']:.3f} "
      f"fp_blobs={m['fp_blobs']}")
