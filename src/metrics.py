

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7


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
    """binary erosion,用 -maxpool(-x) 實作,避免依賴 scipy。"""
    pad = k // 2
    return -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=pad)


def dilate(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)



def iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float((inter + EPS) / (union + EPS))


def dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    inter = (pred * gt).sum()
    return float((2 * inter + EPS) / (pred.sum() + gt.sum() + EPS))


def boundary_iou(pred: torch.Tensor, gt: torch.Tensor, d: int = 2) -> float:
    """
    只在距離邊界 d 個 pixel 的 band 內計算 IoU。

    d 的選擇:原論文用影像對角線的 2%,但 CD-SEM 的線寬本身就窄,
    2% 會把整條線都吃進去。這裡預設 d=2 px,並建議在 paper 中同時報 d=1,2,4。
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    k = 2 * d + 1
    pred_b = pred - erode(pred, k)
    gt_b = gt - erode(gt, k)
    inter = (pred_b * gt_b).sum()
    union = pred_b.sum() + gt_b.sum() - inter
    return float((inter + EPS) / (union + EPS))


def edge_placement_error(pred: torch.Tensor, gt: torch.Tensor,
                         max_search: int = 16) -> dict:
    """
    對 GT 邊界上的每個 pixel,找最近的 pred 邊界 pixel,回傳距離統計。

    以 dilation 逐步擴張近似 distance transform(不需 scipy),
    max_search 之外的視為 max_search(避免離群值主導平均)。

    回傳 {epe_mean, epe_p95}(單位:pixel)
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    gt_b = (gt - erode(gt, 3)) > 0.5
    pred_b = (pred - erode(pred, 3)) > 0.5

    if gt_b.sum() == 0:
        return {"epe_mean": 0.0, "epe_p95": 0.0}

    dist = torch.full_like(gt, float(max_search))
    reached = pred_b.float()
    dist[(reached > 0.5) & gt_b] = 0.0
    cur = reached
    for r in range(1, max_search + 1):
        cur = dilate(cur, 3)
        newly = (cur > 0.5) & (dist >= max_search) & gt_b
        dist[newly] = float(r)
        if not (gt_b & (dist >= max_search)).any():
            break

    vals = dist[gt_b]
    return {
        "epe_mean": float(vals.mean()),
        "epe_p95": float(torch.quantile(vals, 0.95)) if vals.numel() > 1 else float(vals.mean()),
    }



def cd_error(pred: torch.Tensor, gt: torch.Tensor, axis: int = -1) -> dict:
    """
    以 row-wise(或 column-wise)前景總寬度近似 CD。

    對 line/space pattern 而言,每一 row 的前景 pixel 數就是該 row 的
    total line width;pred 與 GT 的差即為 CD error。
    這是最穩健的近似(不需 line instance 對應),適合放進 paper。

    axis=-1: 沿 width 方向數(適合垂直線條);axis=-2 則相反。
    回傳 {cd_mae, cd_bias}。cd_bias 有號,可看出系統性偏胖/偏瘦。
    """
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    p = pred.sum(dim=axis)
    g = gt.sum(dim=axis)
    diff = p - g
    return {"cd_mae": float(diff.abs().mean()), "cd_bias": float(diff.mean())}



def _label_components(mask: torch.Tensor, iters: int = 128) -> torch.Tensor:
    """
    以 max-pool 標籤傳播做 connected component labeling。
    純 torch,不需 scipy;對稀疏的雜訊 blob 收斂很快。
    """
    b, _, h, w = mask.shape
    idx = torch.arange(h * w, device=mask.device, dtype=torch.float32).view(1, 1, h, w)
    lab = idx * mask
    for _ in range(iters):
        new = F.max_pool2d(lab, 3, stride=1, padding=1) * mask
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def _fp_blobs_scipy(fp_np, min_area: int):
    """scipy 的 union-find labeling,比 max-pool 傳播快兩個數量級。"""
    from scipy import ndimage
    n = 0
    for b in range(fp_np.shape[0]):
        lab, k = ndimage.label(fp_np[b, 0])
        if k == 0:
            continue
        counts = ndimage.sum_labels(fp_np[b, 0], lab, index=range(1, k + 1))
        n += int((counts >= min_area).sum())
    return n


def fp_blob_count(pred: torch.Tensor, gt: torch.Tensor, min_area: int = 4) -> dict:
    """
    背景區(GT=0)中,面積 >= min_area 的獨立 false-positive 連通元件個數。

    這個數字直接對應學長講的「noisy back area problem」。
    Phase 2 的成功與否主要看它,而不是 IoU。

    有 scipy 就走 scipy(快兩個數量級);沒有則退回純 torch 的標籤傳播。
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
        uniq, cnt = torch.unique(v, return_counts=True)
        n += int((cnt >= min_area).sum())
    return {"fp_blobs": n, "fp_area": float(fp.sum())}


# ----------------------------------------------------------------------
# 邊界環帶指標 —— Phase 3 用
# ----------------------------------------------------------------------
def border_band_metrics(pred: torch.Tensor, gt: torch.Tensor, band: int = 32) -> dict:
    """
    只在影像最外圈 band 個 pixel 的環帶上計算 IoU / Boundary IoU。

    Phase 3 的 crop-train/full-infer 改善的正是這一圈;
    若混進中央區的大量 pixel 一起算,改善會被稀釋到看不見。

    **注意**:必須先在完整影像上求 boundary,再限制到環帶。
    若先遮罩再求 boundary,環帶內側會產生一圈人造邊緣,而 pred 與 gt
    被切在完全相同的位置、必然吻合,會把 boundary IoU 灌高。
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

    return {
        "border_iou": float((inter + EPS) / (union + EPS)),
        "border_biou": float((bi + EPS) / (bu + EPS)),
    }


# ----------------------------------------------------------------------
# 一次算完
# ----------------------------------------------------------------------
def compute_all(pred: torch.Tensor, gt: torch.Tensor,
                border_band: int = 32, min_blob_area: int = 4,
                light: bool = False) -> dict:
    """
    回傳 paper Table 需要的全部欄位。pred 必須已二值化。

    light=True 時只算便宜的指標(約 20 ms 而非 140 ms),供訓練中每個 epoch 使用;
    昂貴的 EPE / fp_blobs 每 N 個 epoch 或最終評估時才算完整版。
    在 512x512 上,完整版約 140 ms/張 —— 162 張的 val set 每 epoch 就要 23 秒,
    對小模型而言會比訓練本身還久。
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
    pred = gt.clone()
    pred[..., 20:22] = 0                       # 邊緣往內縮 2 px
    pred[0, 0, 5:8, 100:103] = 1               # 背景一個 3x3 假陽性 blob
    for k, v in compute_all(pred, gt).items():
        print(f"{k:>14}: {v}")
