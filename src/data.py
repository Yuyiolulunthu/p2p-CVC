
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN = re.compile(r"([A-Za-z]+)(\d+)")

# 量太少的 family 併成一組，否則 fold 之間不可比。
# dense space / isolated space / isolated line 物理上同屬「孤立或密集的單一特徵」。
FAMILY_MERGE = {"DS": "ISO", "IS": "ISO", "IL": "ISO"}


def family_of(stem: str) -> str:
    f = "".join(sorted(k.upper() for k, _ in TOKEN.findall(stem.replace("_DOM", ""))))
    return FAMILY_MERGE.get(f, f)


def load_pairs(pairs_json: str | Path) -> list[dict]:
    d = json.loads(Path(pairs_json).read_text(encoding="utf-8"))
    return [{"sem": p["sem"], "dom": p["dom"], "stem": p["stem"],
             "family": family_of(p["stem"])} for p in d["pairs"]]


# ----------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------
def make_split(pairs, seed=0, val_ratio=0.15, test_ratio=0.15, out=None):
    """依 family 分層抽樣，確保每個 family 在 train/val/test 都有。"""
    rng = random.Random(seed)
    by_fam: dict[str, list[dict]] = {}
    for it in pairs:
        by_fam.setdefault(it["family"], []).append(it)

    split = {"train": [], "val": [], "test": [], "seed": seed}
    for fam, items in sorted(by_fam.items()):
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_te = max(1, round(n * test_ratio))
        n_va = max(1, round(n * val_ratio))
        split["test"] += [i["stem"] for i in items[:n_te]]
        split["val"] += [i["stem"] for i in items[n_te:n_te + n_va]]
        split["train"] += [i["stem"] for i in items[n_te + n_va:]]

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def lofo_split(pairs, held_out_family: str, seed=0, val_ratio=0.15):
    """
    Leave-One-Family-Out：某個 pattern family 完全不參與訓練，只當 test。

    紀律：val 只能從其餘 family 切。用被留下的 family 調任何東西
    （包含 early stopping），這個 generalization 實驗就無效。
    """
    rng = random.Random(seed)
    tr = [p for p in pairs if p["family"] != held_out_family]
    te = [p for p in pairs if p["family"] == held_out_family]
    assert te, f"family {held_out_family} not found"
    rng.shuffle(tr)
    n_va = max(1, round(len(tr) * val_ratio))
    return {"train": [p["stem"] for p in tr[n_va:]],
            "val": [p["stem"] for p in tr[:n_va]],
            "test": [p["stem"] for p in te],
            "held_out": held_out_family}


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def _gray(path: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


class P2PDataset(Dataset):
    """
    mode='crop'：隨機裁 crop_size（訓練）。影像只有 512×512，
                 所以 Phase 3 的「crop 訓練 / full 推論」是 crop 256 -> infer 512。
    mode='full'：整張 512×512（驗證 / 推論）。

    極性：build_index 已驗證 DOM 白區對應 SEM 亮區，故 mask = (DOM > 0.5)。
          換資料時要重新驗一次，不要沿用這個結論。
    """

    def __init__(self, stems, pairs, crop_size=256, mode="crop",
                 augment=True, crop_margin=0, invert_mask=False):
        idx = {p["stem"]: p for p in pairs}
        self.items = [idx[s] for s in stems if s in idx]
        self.crop_size = crop_size
        self.mode = mode
        self.augment = augment and mode == "crop"
        self.crop_margin = crop_margin
        self.invert_mask = invert_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = _gray(it["sem"])
        msk = (_gray(it["dom"]) > 0.5).astype(np.float32)
        if self.invert_mask:
            msk = 1.0 - msk

        if self.mode == "crop":
            m, cs = self.crop_margin, self.crop_size
            h, w = img.shape
            hi_y, hi_x = max(m, h - m - cs), max(m, w - m - cs)
            y = np.random.randint(m, hi_y + 1) if hi_y > m else 0
            x = np.random.randint(m, hi_x + 1) if hi_x > m else 0
            img, msk = img[y:y + cs, x:x + cs], msk[y:y + cs, x:x + cs]

            if self.augment:
                # 只做幾何 augmentation。刻意不做亮度/對比 jitter：
                # threshold head 依賴原始灰階的物理意義，而本資料的
                # 前景/背景平均亮度差僅約 0.04，任何亮度擾動都會把它淹掉。
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
                "stem": it["stem"], "family": it["family"]}


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--out", default="splits/split_seed0.json")
    a = ap.parse_args()

    pairs = load_pairs(a.pairs)
    print(f"pairs: {len(pairs)}")
    for k, v in Counter(p["family"] for p in pairs).most_common():
        print(f"   {k:<10} {v:>5}")
    sp = make_split(pairs, seed=0, out=a.out)
    print(f"\ntrain/val/test = {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}")
    print(f"saved -> {a.out}")
