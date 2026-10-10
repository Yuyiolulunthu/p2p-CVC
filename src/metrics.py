"""
CD-SEM segmentation metrics.

除了標準 IoU / Dice，這裡實作四個對本任務真正重要的指標：

  boundary_iou          region IoU 對邊緣抖動不敏感，這個才量得到「抓邊緣」
  edge_placement_error  半導體實際的 spec，**雙向**（見下方說明）
  cd_error              CD-SEM 的 downstream metric，line width 誤差
  fp_blob_count         直接量背景的假陽性團塊數

輸入慣例：pred / gt 為 (B,1,H,W) 或 (B,H,W) 的 0/1 tensor。
pred 必須已經二值化（用 binarize()）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7


# ----------------------------------------------------------------------
# 基礎工具
# ----------------------------------------------------------------------
def _to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[:, None]
    return x.float()


def binarize(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """logits -> 0/1。注意輸入是 logit 不是機率。"""
    return (torch.sigmoid(logits) > thr).float()


def erode(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """binary erosion，用 -maxpool(-x) 實作，避免依賴 scipy。"""
    return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)


def dilate(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)


# ----------------------------------------------------------------------
# 標準指標
# ----------------------------------------------------------------------
def iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float((inter + EPS) / (union + EPS))


def dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    inter = (pred * gt).sum()
    return float((2 * inter + EPS) / (pred.sum() + gt.sum() + EPS))


# ----------------------------------------------------------------------
# Boundary IoU (Cheng et al., CVPR 2021)
# ----------------------------------------------------------------------
def boundary_iou(pred: torch.Tensor, gt: torch.Tensor, d: int = 2) -> float:
    """
    只在距離邊界 d 個 pixel 的 band 內計算 IoU。

    原論文用影像對角線的 2%，但 CD-SEM 的線寬本身就窄，2% 會把整條線吃進去。
    這裡預設 d=2 px，建議論文同時報 d=1,2,4。
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    k = 2 * d + 1
    pred_b = pred - erode(pred, k)
    gt_b = gt - erode(gt, k)
    inter = (pred_b * gt_b).sum()
    union = pred_b.sum() + gt_b.sum() - inter
    return float((inter + EPS) / (union + EPS))


# ----------------------------------------------------------------------
# Edge Placement Error（雙向）
# ----------------------------------------------------------------------
def _dist_to(target_edge: torch.Tensor, query_edge: torch.Tensor,
             max_search: int) -> torch.Tensor:
    """
    對 query_edge 上的每個像素，求到最近 target_edge 像素的距離。
    以逐步膨脹近似距離轉換（不需 scipy）；超過 max_search 的視為 max_search，
    避免少數離群值主導平均。
    """
    dist = torch.full(query_edge.shape, float(max_search),
                      device=query_edge.device, dtype=torch.float32)
    cur = target_edge.float()
    dist[(cur > 0.5) & query_edge] = 0.0
    for r in range(1, max_search + 1):
        if not (query_edge & (dist >= max_search)).any():
            break
        cur = dilate(cur, 3)
        newly = (cur > 0.5) & (dist >= max_search) & query_edge
        dist[newly] = float(r)
    return dist[query_edge]


def edge_placement_error(pred: torch.Tensor, gt: torch.Tensor,
                         max_search: int = 16) -> dict:
    """
    雙向 edge placement error（單位：pixel）。

    單向版本（只算 GT 邊界 -> 最近預測邊界）可以作弊：預測如果到處都是邊緣，
    任何 GT 邊界附近都剛好有一個預測邊界，分數就趨近 0。古典 thresholding
    產生一千多個雜訊團塊卻拿到 EPE 0.37，就是這個漏洞。

    改成雙向之後：
      epe_gt2pred  GT 邊界 -> 最近預測邊界    漏掉真實邊緣會被罰（召回面）
      epe_pred2gt  預測邊界 -> 最近 GT 邊界    亂畫邊緣會被罰（精確面）
      epe_mean     兩者平均（對稱）           <- 論文報這個
      epe_p95      兩個方向合起來的 95 百分位
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    gt_b = (gt - erode(gt, 3)) > 0.5
    pred_b = (pred - erode(pred, 3)) > 0.5

    # 任一側沒有邊界時距離無從定義。回傳 max_search 當最差值，而不是 0 ——
    # 全白或全黑的預測不該被當成完美。
    if not gt_b.any() or not pred_b.any():
        worst = float(max_search)
        return {"epe_mean": worst, "epe_p95": worst,
                "epe_gt2pred": worst, "epe_pred2gt": worst}

    d1 = _dist_to(pred_b, gt_b, max_search)      # GT   -> pred
    d2 = _dist_to(gt_b, pred_b, max_search)      # pred -> GT
    both = torch.cat([d1, d2])

    return {
        "epe_mean": float((d1.mean() + d2.mean()) / 2),
        "epe_p95": float(torch.quantile(both, 0.95)) if both.numel() > 1
                   else float(both.mean()),
        "epe_gt2pred": float(d1.mean()),
        "epe_pred2gt": float(d2.mean()),
    }


# ----------------------------------------------------------------------
# CD (Critical Dimension) error —— domain-specific，建議一定要報
# ----------------------------------------------------------------------
def cd_error(pred: torch.Tensor, gt: torch.Tensor, axis: int = -1) -> dict:
    """
    以 row-wise 前景總寬度近似 CD。

    對 line/space pattern 而言，每一 row 的前景 pixel 數就是該 row 的
    total line width；pred 與 GT 的差即為 CD error。不需要 line instance
    對應，很穩健。

    cd_bias 有號，可看出系統性偏胖/偏瘦；cd_mae 則反映 line edge roughness。
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    diff = pred.sum(dim=axis) - gt.sum(dim=axis)
    return {"cd_mae": float(diff.abs().mean()), "cd_bias": float(diff.mean())}


# ----------------------------------------------------------------------
# 背景 false-positive blob 計數
# ----------------------------------------------------------------------
def _label_components(mask: torch.Tensor, iters: int = 128) -> torch.Tensor:
    """
    純 torch 的 connected component labeling（max-pool 標籤傳播）。
    只在沒有 scipy 時當後備：雜訊多時可能在 iters 內收斂不完，
    會把同一個元件拆成多個，所以務必裝 scipy。
    """
    b, _, h, w = mask.shape
    idx = torch.arange(h * w, device=mask.device,
                       dtype=torch.float32).view(1, 1, h, w)
    lab = idx * mask
    for _ in range(iters):
        new = F.max_pool2d(lab, 3, stride=1, padding=1) * mask
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def _fp_blobs_scipy(fp_np, min_area: int) -> int:
    """scipy 的 union-find labeling，比 max-pool 傳播快兩個數量級。"""
    from scipy import ndimage
    n = 0
    for b in range(fp_np.shape[0]):
        lab, k = ndimage.label(fp_np[b, 0])
        if k == 0:
            continue
        counts = ndimage.sum_labels(fp_np[b, 0], lab, index=range(1, k + 1))
        n += int((counts >= min_area).sum())
    return n


def fp_blob_count(pred: torch.Tensor, gt: torch.Tensor,
                  min_area: int = 4) -> dict:
    """
    背景區（GT=0）中，面積 >= min_area 的獨立 false-positive 連通元件個數。
    直接對應「noisy background」問題。
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    fp = ((pred > 0.5) & (gt < 0.5)).float()
    if fp.sum() == 0:
        return {"fp_blobs": 0, "fp_area": 0.0}

    try:
        return {"fp_blobs": _fp_blobs_scipy(fp.detach().cpu().numpy(), min_area),
                "fp_area": float(fp.sum())}
    except ImportError:
        pass

    lab = _label_components(fp)
    n = 0
    for bi in range(lab.shape[0]):
        v = lab[bi][fp[bi] > 0.5]
        if v.numel() == 0:
            continue
        _, cnt = torch.unique(v, return_counts=True)
        n += int((cnt >= min_area).sum())
    return {"fp_blobs": n, "fp_area": float(fp.sum())}


# ----------------------------------------------------------------------
# 邊界環帶指標
# ----------------------------------------------------------------------
def border_band_metrics(pred: torch.Tensor, gt: torch.Tensor,
                        band: int = 32) -> dict:
    """
    只在影像最外圈 band 個 pixel 的環帶上計算 IoU / Boundary IoU。

    **必須先在完整影像上求 boundary，再限制到環帶。** 若先遮罩再求 boundary，
    環帶內側會產生一圈人造邊緣，而 pred 與 gt 被切在完全相同的位置、必然吻合，
    會把 boundary IoU 灌高。
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    _, _, h, w = pred.shape
    m = torch.ones_like(pred)
    m[..., band:h - band, band:w - band] = 0

    inter = (pred * gt * m).sum()
    union = (pred * m).sum() + (gt * m).sum() - inter

    k = 5   # d=2
    pred_b = (pred - erode(pred, k)) * m
    gt_b = (gt - erode(gt, k)) * m
    bi = (pred_b * gt_b).sum()
    bu = pred_b.sum() + gt_b.sum() - bi

    return {"border_iou": float((inter + EPS) / (union + EPS)),
            "border_biou": float((bi + EPS) / (bu + EPS))}


# ----------------------------------------------------------------------
# 一次算完
# ----------------------------------------------------------------------
def compute_all(pred: torch.Tensor, gt: torch.Tensor,
                border_band: int = 32, min_blob_area: int = 4,
                light: bool = False) -> dict:
    """
    回傳 paper Table 需要的全部欄位。pred 必須已二值化。

    light=True 只算便宜的指標（約 20 ms 而非 140 ms），供訓練中每個 epoch 使用；
    昂貴的 EPE / fp_blobs 每 N 個 epoch 或最終評估時才算完整版。
    """
    out = {"iou": iou(pred, gt), "dice": dice(pred, gt),
           "biou_d2": boundary_iou(pred, gt, d=2)}
    if light:
        return out
    for d in (1, 4):
        out[f"biou_d{d}"] = boundary_iou(pred, gt, d=d)
    out.update(edge_placement_error(pred, gt))
    out.update(cd_error(pred, gt))
    out.update(fp_blob_count(pred, gt, min_area=min_blob_area))
    out.update(border_band_metrics(pred, gt, band=border_band))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    gt = torch.zeros(1, 1, 128, 128)
    gt[..., 20:40] = 1
    gt[..., 70:90] = 1

    print("--- 邊緣內縮 2px + 一個背景雜訊團塊 ---")
    pred = gt.clone()
    pred[..., 20:22] = 0
    pred[0, 0, 5:8, 100:103] = 1
    for k, v in compute_all(pred, gt).items():
        print(f"{k:>14}: {v}")

    print("\n--- 亂畫邊緣（20% 雜訊）：舊版單向 EPE 會給接近滿分 ---")
    noisy = ((gt + (torch.rand_like(gt) < 0.2).float()) > 0).float()
    e = edge_placement_error(noisy, gt)
    print(f"  epe_gt2pred={e['epe_gt2pred']:.3f}  (單向舊版就是報這個，看起來很好)")
    print(f"  epe_pred2gt={e['epe_pred2gt']:.3f}  (亂畫的邊緣離 GT 很遠)")
    print(f"  epe_mean   ={e['epe_mean']:.3f}  (雙向，無法作弊)")