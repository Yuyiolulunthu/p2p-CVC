"""
Loss —— 對齊學長 main.py 的 loss_fn。

他的寫法：
    bce  = F.binary_cross_entropy_with_logits(pred, gt)
    dice = 1 - (2*(sigmoid(pred)*gt).sum() + eps) / (sigmoid(pred).sum() + gt.sum() + eps)
    return bce + 0.5 * dice

兩點與他不同，各自寫明理由：

1. 權重沿用他的 1.0 * BCE + 0.5 * Dice（我原本是 0.5/0.5）。要和他的
   baseline 數字對得起來，loss 必須一樣。

2. Dice 改成 per-image 後取平均，他是整個 batch 一起 sum。他的 BATCH_SIZE=1，
   兩者完全等價；但我們為了記憶體會用不同的 batch size 跑不同 encoder，
   batch 級的 sum 會讓 loss 隨 batch size 改變，那樣跨 encoder 的比較就不公平。
   per-image 平均與 batch size 無關。要完全複刻他的行為時傳 dice_reduction="global"。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.threshold_head import total_variation

EPS = 1e-6


def dice_loss(logits, target, reduction="per_image"):
    p = torch.sigmoid(logits)
    if reduction == "global":                      # 學長原始寫法
        num = 2 * (p * target).sum() + EPS
        den = p.sum() + target.sum() + EPS
        return 1 - num / den
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + EPS
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + EPS
    return (1 - num / den).mean()


def seg_loss(out: dict, target: torch.Tensor,
             lambda_tv: float = 0.0, lambda_r: float = 0.0,
             w_bce: float = 1.0, w_dice: float = 0.5,
             dice_reduction: str = "per_image") -> tuple:
    logits = out["logits"]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dsc = dice_loss(logits, target, dice_reduction)
    total = w_bce * bce + w_dice * dsc
    parts = {"bce": bce.item(), "dice": dsc.item()}

    # --- Phase 2 專用的兩項，預設關閉 ---
    if lambda_tv > 0 and out.get("threshold") is not None:
        tv = total_variation(out["threshold"])
        total = total + lambda_tv * tv
        parts["tv"] = tv.item()

    if lambda_r > 0 and out.get("residual") is not None:
        r = out["residual"].abs().mean()
        total = total + lambda_r * r
        parts["r_l1"] = r.item()      # 多少決策必須繞過 threshold 機制

    parts["total"] = total.item()
    return total, parts