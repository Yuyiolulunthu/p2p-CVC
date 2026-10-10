"""
統一的模型建構入口。

公平比較的前提:**decoder 完全固定,只換 encoder**。
所以任何實驗都只能透過 build_model() 取得模型,不要在別處自己組。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .threshold_head import ThresholdHead, total_variation  # noqa: F401

DECODER_CHANNELS = (256, 128, 64, 32, 16)   # 全專案固定,不准 per-model 調


class SegModel(nn.Module):
    """smp U-Net(去掉原本的 segmentation_head)+ 我們自己的 ThresholdHead。"""

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_ch=1, variant="v0", k_init=10.0, window=31):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=1,
            decoder_channels=DECODER_CHANNELS,
            activation=None,
        )
        self.net.segmentation_head = nn.Identity()   # 由 ThresholdHead 取代
        self.head = ThresholdHead(DECODER_CHANNELS[-1], variant=variant,
                                  k_init=k_init, window=window)
        self.variant = variant

    def forward(self, x, raw=None):
        """
        x   : 餵給 encoder 的張量(可經過正規化)
        raw : 原始 [0,1] 灰階影像,threshold head 要用。
              預設等於 x —— 若 dataset 有做 mean/std 正規化,**必須另外傳 raw**,
              否則 threshold 會失去物理意義。
        """
        feat = self.net(x)                 # (B, 16, H, W)
        return self.head(feat, x if raw is None else raw)


def build_model(cfg) -> SegModel:
    return SegModel(
        encoder_name=cfg["encoder"],
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_ch=cfg.get("in_channels", 1),
        variant=cfg.get("variant", "v0"),
        k_init=cfg.get("k_init", 10.0),
        window=cfg.get("threshold_window", 31),
    )


def count_params(m: nn.Module) -> float:
    """百萬(M)為單位。"""
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6


@torch.no_grad()
def resolution_report(model: SegModel, sizes=(256, 384, 512, 768, 1024),
                      in_ch=1, device="cpu") -> dict:
    """
    Phase 3 可行性前測:同一組權重能不能吃不同解析度?

    mit_b0 應該全過(無 positional encoding);
    在此掛掉的 encoder,phase 3 就要特別處理 —— 這張表直接進 paper Table 2。
    """
    model = model.to(device).eval()
    out = {}
    for s in sizes:
        try:
            y = model(torch.randn(1, in_ch, s, s, device=device))["logits"]
            out[s] = f"OK {tuple(y.shape)}"
        except Exception as e:                      # noqa: BLE001
            out[s] = f"FAIL {type(e).__name__}"
    return out
