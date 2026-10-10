"""

逐段計時，指出訓練慢在哪裡：裝置、資料載入、還是前向/反向。
"""

import json
import time

import torch
from torch.utils.data import DataLoader

from src.data import P2PDataset, load_pairs
from src.models import build_model, count_params

# ---------- 1. 裝置 ----------
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda, "  <- None 代表裝到 CPU 版")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=> 將使用 {dev}\n")

# ---------- 2. 資料載入 ----------
pairs = load_pairs("pairs.json")
split = json.load(open("splits/split_seed0.json", encoding="utf-8"))
ds = P2PDataset(split["train"], pairs, crop_size=256, mode="crop", augment=True)
print(f"train set: {len(ds)}")

t = time.perf_counter()
n = min(20, len(ds))
for i in range(n):
    _ = ds[i]
dt = (time.perf_counter() - t) / n
print(f"單張讀取+augment  {dt*1000:6.1f} ms  -> 759 張/epoch = {dt*759:5.1f} s")

dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
t = time.perf_counter()
for i, b in enumerate(dl):
    if i >= 5:
        break
print(f"DataLoader 5 個 batch  {(time.perf_counter()-t):5.2f} s\n")

# ---------- 3. 前向 / 反向 ----------
model = build_model({"encoder": "resnet34", "encoder_weights": "imagenet",
                     "in_channels": 1, "variant": "v0"}).to(dev)
print(f"params: enc={count_params(model.net.encoder):.2f}M total={count_params(model):.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

x = torch.randn(8, 1, 256, 256, device=dev)
y = (torch.rand(8, 1, 256, 256, device=dev) > 0.5).float()

# warmup（第一次會觸發 cudnn autotune，特別慢，不列入計時）
t = time.perf_counter()
for _ in range(3):
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x)["logits"], y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
if dev == "cuda":
    torch.cuda.synchronize()
print(f"warmup 3 iters  {time.perf_counter()-t:5.2f} s")

t = time.perf_counter()
N = 10
for _ in range(N):
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x)["logits"], y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
if dev == "cuda":
    torch.cuda.synchronize()
it = (time.perf_counter() - t) / N
print(f"每個 iteration   {it*1000:6.1f} ms")

n_iter = len(ds) // 8
print(f"\n=> 純計算 {n_iter} iters/epoch = {it*n_iter:5.1f} s")
print(f"=> 加上資料載入約 {it*n_iter + dt*759:5.1f} s/epoch")
print(f"=> 100 epochs ≈ {(it*n_iter + dt*759)*100/60:5.1f} 分鐘（不含驗證）")
print("\n驗證另計：light eval 約 3 s，full eval 約 21 s（每 10 epoch 一次）")
