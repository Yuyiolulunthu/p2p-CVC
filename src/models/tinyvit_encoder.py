"""
TinyViT (MobileSAM / LiteMedSAM) encoder wrapper。

這是三個候選裡唯一有真實工程成本的,**第一天就要開始寫,不要留到最後一週**。

步驟:
  1. git clone https://github.com/ChaoningZhang/MobileSAM
     或 MedSAM 的 LiteMedSAM branch(拿 medical distilled 權重 —— 這才是
     學長「maybe on medical images」真正指的東西,也是我們 domain-gap 論點的來源)
  2. 用 probe_stages() 印出實際的 feature shape
  3. 把 stride 序列與 channel 填進 register_tinyvit()

**已知陷阱**:MobileSAM 為了對齊 SAM 的輸出解析度,把 TinyViT 最後一次
downsample 的 stride 由 2 改成 1。所以實際 stride 序列 **不是** [4,8,16,32]。
不要照抄標準 hierarchical backbone 的假設 —— 先印出來看。

    python -m src.models.tinyvit_encoder --ckpt mobile_sam.pt --probe
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyViTEncoder(nn.Module):
    """
    包成 smp EncoderMixin 相容介面:forward 回傳 feature list,
    index 0 為最高解析度(smp decoder 的期待)。
    """

    def __init__(self, tinyvit: nn.Module, stages, out_channels, depth=5, in_ch=1):
        super().__init__()
        self.model = tinyvit
        self._out_channels = tuple(out_channels)
        self._depth = depth
        self._in_channels = in_ch
        self._feats: list[torch.Tensor] = []
        for s in stages:
            s.register_forward_hook(self._hook)

    def _hook(self, m, i, o):
        # 有些實作輸出是 (B, N, C) 的 token,要轉回 (B, C, H, W)
        if o.dim() == 3:
            b, n, c = o.shape
            hw = int(n ** 0.5)
            o = o.transpose(1, 2).reshape(b, c, hw, hw)
        self._feats.append(o)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)     # SAM 權重是三通道;灰階複製三份最安全
        self._feats = []
        self.model(x)
        # smp Unet 需要 depth+1 個 feature(含 identity 的 stage0)
        return [x] + self._feats

    def set_in_channels(self, in_channels, pretrained=True):
        self._in_channels = in_channels   # 我們在 forward 內處理,不改權重

    def make_dilated(self, *a, **k):
        raise NotImplementedError("TinyViT wrapper 不支援 dilation")


@torch.no_grad()
def probe_stages(tinyvit: nn.Module, size=512, in_ch=3, device="cpu"):
    """
    印出每個子模組的輸出 shape,用來決定要 hook 哪四個 stage、
    以及實際的 stride 序列。**接 decoder 之前一定要先跑這個。**
    """
    shapes = []
    hooks = []

    def mk(name):
        def f(m, i, o):
            if torch.is_tensor(o):
                shapes.append((name, tuple(o.shape)))
        return f

    for n, m in tinyvit.named_children():
        hooks.append(m.register_forward_hook(mk(n)))
    tinyvit.to(device).eval()(torch.randn(1, in_ch, size, size, device=device))
    for h in hooks:
        h.remove()
    for n, s in shapes:
        stride = size // s[-1] if len(s) == 4 else "-"
        print(f"{n:<24} {str(s):<28} stride={stride}")
    return shapes


def load_mobilesam_tinyvit(ckpt_path: str):
    """
    需要 MobileSAM repo 在 PYTHONPATH。
    LiteMedSAM 的權重結構相同,只是 pretrain domain 不同 —— 兩個都要跑,
    因為「ImageNet vs medical pretrain」的比較正是我們 paper 的一段論述。
    """
    from mobile_sam import sam_model_registry
    sam = sam_model_registry["vit_t"](checkpoint=ckpt_path)
    return sam.image_encoder


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    enc = load_mobilesam_tinyvit(a.ckpt)
    if a.probe:
        probe_stages(enc, size=a.size)
