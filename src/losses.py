

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models.threshold_head import total_variation


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def seg_loss(out: dict, target: torch.Tensor,
             lambda_tv: float = 0.0, lambda_r: float = 0.0) -> tuple:
    logits = out["logits"]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dsc = dice_loss(logits, target)
    total = 0.5 * bce + 0.5 * dsc
    parts = {"bce": bce.item(), "dice": dsc.item()}

    if lambda_tv > 0 and out.get("threshold") is not None:
        tv = total_variation(out["threshold"])
        total = total + lambda_tv * tv
        parts["tv"] = tv.item()

    if lambda_r > 0 and out.get("residual") is not None:
        r = out["residual"].abs().mean()
        total = total + lambda_r * r
        parts["r_l1"] = r.item()     # 這個數字要記錄:多少像素需要繞過 threshold 機制

    parts["total"] = total.item()
    return total, parts
