"""
Phase 2 —— Adaptive Thresholding Head。

核心想法:
    不讓 decoder 直接吐 binary logit,而是吐一張 per-pixel threshold field T(x),
    最終決策綁回原始 SEM 的實際灰階:

        logit(x) = k(x) * ( I(x) - T(x) )

    低 SNR 的背景區,T 是平滑的 -> 雜訊點的灰階跨不過門檻 -> 不會產生孤立 blob。
    這是 arXiv:2306.14250「thresholding 應是 adaptive 而非 static」的可學習版本。

五個變體(即 Phase 2 的 ablation table):
    V0  direct        logit = f(x)                            現行 baseline
    V1  threshold     logit = k*(I-T),      k 為常數
    V2  thr+gain      logit = k(x)*(I-T(x)) k 也可學
    V3  thr+residual  logit = k(x)*(I-T(x)) + r(x),r 上 L1    安全版
    V4  local-mean    T = LocalMean(I,w) + dT(x),dT 初始化為 0  ★ 推薦

V4 的關鍵:初始化時 dT=0,整個模型退化成傳統 adaptive mean thresholding
(Niblack / Sauvola 家族),所以起點保證不比古典方法差,decoder 只學殘差。
paper 裡的說法:「我們把古典 adaptive thresholding 當作 inductive bias。」
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("v0", "v1", "v2", "v3", "v4")


def local_mean(x: torch.Tensor, w: int = 31) -> torch.Tensor:
    """以 reflect padding 的 box filter 求局部平均,邊界不會被 0 拉低。"""
    pad = w // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=w, stride=1)


class ThresholdHead(nn.Module):
    """
    接在 U-Net decoder 之後。輸入 decoder feature 與原始影像,輸出 logit。

    參數
    ----
    in_ch    : decoder 最後一層的 channel 數
    variant  : v0..v4
    k_init   : 陡峭度初始值。太小會讓梯度太平,太大會讓早期訓練不穩;
               預設 10.0 對應「灰階差 0.1 -> logit 1.0」,對 [0,1] 正規化的 SEM 合適。
    window   : v4 的局部平均視窗。應略大於 line pitch,否則 local mean 會被線條本身帶走。
    """

    def __init__(self, in_ch: int, variant: str = "v4",
                 k_init: float = 10.0, window: int = 31):
        super().__init__()
        assert variant in VARIANTS, f"unknown variant {variant}"
        self.variant = variant
        self.window = window

        if variant == "v0":
            self.head = nn.Conv2d(in_ch, 1, 1)
            return

        # 預測 threshold(或其殘差)
        self.t_head = nn.Conv2d(in_ch, 1, 1)
        # 預測 gain
        self.k_head = nn.Conv2d(in_ch, 1, 1) if variant in ("v2", "v3") else None
        # 預測 residual logit
        self.r_head = nn.Conv2d(in_ch, 1, 1) if variant == "v3" else None

        self.log_k = nn.Parameter(torch.tensor(float(k_init)).log())

        if variant == "v4":
            # dT 初始化為 0 -> 起點等價於古典 adaptive mean threshold
            nn.init.zeros_(self.t_head.weight)
            nn.init.zeros_(self.t_head.bias)
        else:
            # 其餘變體把 T 初始化在中灰階附近
            nn.init.zeros_(self.t_head.weight)
            nn.init.constant_(self.t_head.bias, 0.5)

        for h in (self.k_head, self.r_head):
            if h is not None:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

    def forward(self, feat: torch.Tensor, image: torch.Tensor) -> dict:
        """
        feat  : (B,C,H,W) decoder 輸出
        image : (B,1,H,W) 原始 SEM,**必須事先正規化到 [0,1]**
                (不要用 ImageNet mean/std 正規化後的張量,那樣 threshold 沒有物理意義)

        回傳 dict:logits / threshold / gain / residual
        """
        if self.variant == "v0":
            return {"logits": self.head(feat), "threshold": None,
                    "gain": None, "residual": None}

        if image.shape[1] != 1:
            image = image.mean(dim=1, keepdim=True)
        if image.shape[-2:] != feat.shape[-2:]:
            image = F.interpolate(image, size=feat.shape[-2:],
                                  mode="bilinear", align_corners=False)

        if self.variant == "v4":
            base = local_mean(image, self.window)
            thr = base + self.t_head(feat)          # 學殘差
        else:
            thr = self.t_head(feat)

        if self.k_head is not None:
            # softplus 保證 gain 為正,否則前景/背景會被反轉
            gain = F.softplus(self.log_k.exp() + self.k_head(feat))
        else:
            gain = self.log_k.exp()

        logits = gain * (image - thr)

        residual = None
        if self.r_head is not None:
            residual = self.r_head(feat)
            logits = logits + residual

        return {"logits": logits, "threshold": thr,
                "gain": gain if torch.is_tensor(gain) else None,
                "residual": residual}


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """
    threshold field 的平滑性正則。

    這是「解掉 noisy background」的實際機制:
    強迫 T 平滑,雜訊就無法被 T 局部下壓而變成前景。
    lambda_tv 建議掃 {0, 1e-3, 1e-2, 1e-1} -> Table 4。
    """
    if x is None:
        return torch.zeros((), device="cpu")
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


if __name__ == "__main__":
    feat = torch.randn(2, 16, 64, 64)
    img = torch.rand(2, 1, 64, 64)
    for v in VARIANTS:
        out = ThresholdHead(16, variant=v)(feat, img)
        t = out["threshold"]
        print(f"{v}: logits={tuple(out['logits'].shape)} "
              f"T={'-' if t is None else tuple(t.shape)}")
