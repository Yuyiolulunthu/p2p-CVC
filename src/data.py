"""
資料層 —— 對齊學長 sam_binary_code 的實際結構。

    dataset/input/<...>/<name>.jpg    灰階 SEM
    dataset/gt/<...>/<name>.jpg       二值 SEM（thresholding + cleaning 產生的 GT）

配對規則與學長的 SEMDataset 完全相同：**相對路徑相同即為一對**。
不再解析檔名 token，pairs.json 與 build_index.py 都不需要了。

與學長 code 的差異（刻意的，各自寫明理由）：
  1. 掃描後依相對路徑排序再切分。他的 os.walk 順序取決於檔案系統，
     換一台機器就可能不同；排序後任何機器都切出同一份 split。
  2. 多切一個 val。他的 code 只有 train/test，沒有 val 就無法做
     early stopping 或選超參數，而我們要比較多個 encoder，一定要有。
     test 比例與他相同（10%），所以 test 上的數字仍可與他的 baseline 對照。
  3. class 取自 input/ 下的第一層子資料夾（若為扁平結構才退回檔名 token）。
     slide 8 的 leave-one-class-out 就以此為單位。
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TOKEN = re.compile(r"([A-Za-z]+)(\d+)")

# 扁平結構時才會用到：量太少的 family 併成一組
FAMILY_MERGE = {"DS": "ISO", "IS": "ISO", "IL": "ISO"}


def family_of(stem: str) -> str:
    f = "".join(sorted(k.upper() for k, _ in TOKEN.findall(stem)))
    return FAMILY_MERGE.get(f, f)


# ----------------------------------------------------------------------
# 掃描與配對
# ----------------------------------------------------------------------
def _index_dir(root: Path) -> dict[str, list[str]]:
    """走訪一個資料夾，回傳 {相對路徑小寫: [實際相對路徑, ...]}。"""
    idx: dict[str, list[str]] = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                rel = (Path(d) / f).relative_to(root).as_posix()
                idx.setdefault(rel.lower(), []).append(rel)
    return idx


def scan_pairs(root: str | Path, input_dir="input", gt_dir="gt",
               verbose=True) -> list[dict]:
    """
    回傳 [{input, gt, rel, name, cls}, ...]，依 rel 排序（決定性）。

    配對不分大小寫。這份資料裡 input/Proximity_N 對應的是 gt/proximity_N，
    用嚴格比對會靜默掉 77 張。學長的 SEMDataset 用 os.path.exists 判斷，
    在 Windows 上剛好不分大小寫所以沒事，但同一份 code 在 Linux 上會掉檔案 ——
    這裡明確處理掉，並把不一致的項目列出來，兩個平台的行為才會一致。
    """
    root = Path(root)
    in_root, gt_root = root / input_dir, root / gt_dir
    if not in_root.exists() or not gt_root.exists():
        raise FileNotFoundError(f"需要 {in_root} 與 {gt_root}")

    gt_idx = _index_dir(gt_root)
    items, missing, case_diff, ambiguous = [], [], [], []

    for rel_lower, rels in sorted(_index_dir(in_root).items()):
        if len(rels) > 1:                       # 同一路徑只有大小寫不同
            ambiguous.append(rels)
        rel = rels[0]
        cand = gt_idx.get(rel_lower)
        if not cand:
            missing.append(rel)
            continue
        gt_rel = cand[0]
        if gt_rel != rel:
            case_diff.append((rel, gt_rel))
        items.append({"input": str(in_root / rel), "gt": str(gt_root / gt_rel),
                      "rel": rel, "name": Path(rel).name,
                      "cls": rel.split("/")[0] if "/" in rel
                      else family_of(Path(rel).stem)})

    items.sort(key=lambda d: d["rel"])          # 決定性順序
    if verbose:
        print(f"pairs: {len(items)}")
        if case_diff:
            print(f"  大小寫不一致但已配對: {len(case_diff)} 筆，"
                  f"例: {case_diff[0][0]} <-> {case_diff[0][1]}")
        if missing:
            print(f"  ** 找不到 GT: {len(missing)} 筆，例: {missing[:3]}")
        if ambiguous:
            print(f"  ** 僅大小寫不同的重複路徑: {len(ambiguous)} 組，"
                  f"例: {ambiguous[0]}")
    return items


# ----------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------
def make_split(items, seed=1, test_ratio=0.1, val_ratio=0.1,
               out=None, test_names=None):
    """
    依學長的配方切分：先隨機抽 test_ratio 當 test，其餘為 train，
    再從 train 內抽 val_ratio 當 val。seed 預設 1，與他的 main.py 相同。

    test_names: 若學長能提供他那份 test 的檔名清單（相對路徑），傳進來就會
                直接沿用，這樣 test set 與他逐張相同，數字可以逐張對照。
    """
    rels = [it["rel"] for it in items]

    if test_names:
        want = {r.replace("\\", "/") for r in test_names}
        test = [r for r in rels if r in want]
        rest = [r for r in rels if r not in want]
        if len(test) != len(want):
            print(f"  ** test_names 有 {len(want)} 筆，實際對上 {len(test)} 筆")
    else:
        idx = list(range(len(rels)))
        random.Random(seed).shuffle(idx)
        n_te = max(1, int(test_ratio * len(idx)))
        test = [rels[i] for i in idx[:n_te]]
        rest = [rels[i] for i in idx[n_te:]]

    r2 = list(rest)
    random.Random(seed + 1000).shuffle(r2)
    n_va = max(1, int(val_ratio * len(r2)))
    split = {"val": sorted(r2[:n_va]), "train": sorted(r2[n_va:]),
             "test": sorted(test), "seed": seed,
             "from_senior_test_list": bool(test_names)}

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def loco_split(items, held_out_class: str, seed=1, val_ratio=0.1):
    """
    Leave-one-class-out（slide 8 第 4 項）。
    紀律：val 只能從其餘 class 切。用被留下的 class 調任何東西，實驗就無效。
    """
    tr = [it["rel"] for it in items if it["cls"] != held_out_class]
    te = [it["rel"] for it in items if it["cls"] == held_out_class]
    if not te:
        raise ValueError(f"class {held_out_class} 不存在；"
                         f"可用的有 {sorted({i['cls'] for i in items})}")
    random.Random(seed).shuffle(tr)
    n_va = max(1, int(val_ratio * len(tr)))
    return {"train": sorted(tr[n_va:]), "val": sorted(tr[:n_va]),
            "test": sorted(te), "held_out": held_out_class}


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class SEMDataset(Dataset):
    """
    影像處理與學長的 SEMDataset 一致：
      input  雙線性 resize 到 size
      gt     最近鄰 resize 到 size，再以 >0.5 二值化
      augment 只做幾何（hflip / vflip / rot90）

    回傳 x 為單通道；要三通道的 encoder（SAM 系）由 model 端自行 repeat，
    不在這裡做，否則非 SAM 的 encoder 會白吃三倍的第一層計算。
    """

    def __init__(self, rels, items, size=512, augment=False,
                 crop_size=0, crop_margin=0):
        idx = {it["rel"]: it for it in items}
        self.items = [idx[r] for r in rels if r in idx]
        self.size = size
        self.augment = augment
        self.crop_size = crop_size          # 0 = 不裁切（Phase 3 才用）
        self.crop_margin = crop_margin

    def __len__(self):
        return len(self.items)

    def _load(self, path, nearest=False):
        from PIL import Image
        im = Image.open(path).convert("L")
        if im.size != (self.size, self.size):
            im = im.resize((self.size, self.size),
                           Image.NEAREST if nearest else Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0

    def __getitem__(self, i):
        it = self.items[i]
        img = self._load(it["input"])
        msk = (self._load(it["gt"], nearest=True) > 0.5).astype(np.float32)

        if self.crop_size:
            m, cs = self.crop_margin, self.crop_size
            h, w = img.shape
            hi_y, hi_x = max(m, h - m - cs), max(m, w - m - cs)
            y = np.random.randint(m, hi_y + 1) if hi_y > m else 0
            x = np.random.randint(m, hi_x + 1) if hi_x > m else 0
            img, msk = img[y:y + cs, x:x + cs], msk[y:y + cs, x:x + cs]

        if self.augment:
            # 只做幾何變換，與學長一致。不做亮度/對比 jitter：
            # Phase 2 的 threshold head 依賴原始灰階的物理意義。
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)
            img, msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)

        x = torch.from_numpy(img.copy())[None]
        y = torch.from_numpy(msk.copy())[None]
        return {"x": x, "raw": x.clone(), "y": y,
                "rel": it["rel"], "name": it["name"], "cls": it["cls"]}


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="splits/split_seed1.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--test-names", default=None,
                    help="學長那份 test 檔名清單（每行一個相對路徑）")
    a = ap.parse_args()

    items = scan_pairs(a.root)
    print("\nclass 分布:")
    for k, v in sorted(Counter(i["cls"] for i in items).items()):
        print(f"   {k:<24} {v:>5}")

    names = None
    if a.test_names:
        names = [l.strip() for l in Path(a.test_names).read_text(
            encoding="utf-8").splitlines() if l.strip()]

    sp = make_split(items, seed=a.seed, out=a.out, test_names=names)
    print(f"\ntrain/val/test = {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}")
    print(f"saved -> {a.out}")