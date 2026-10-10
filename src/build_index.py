
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
TOKEN = re.compile(r"([A-Za-z]+)(\d+)")


def imgs(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def parse_tokens(stem: str) -> dict:
    """'L105S45LW200' -> {'L':105,'S':45,'LW':200}；順序無關，方便反序比對。"""
    s = stem.replace("_DOM", "")
    return {k.upper(): int(v) for k, v in TOKEN.findall(s)}


def key_of(tok: dict) -> tuple:
    """把 token dict 轉成順序無關的可雜湊 key。"""
    return tuple(sorted(tok.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/p2p/data/5")
    ap.add_argument("--sem-dir", default=None)
    ap.add_argument("--dom-dir", default=None)
    ap.add_argument("--out", default="pairs.json")
    ap.add_argument("--n-check", type=int, default=30, help="極性檢查取樣張數")
    args = ap.parse_args()

    root = Path(args.root)
    subs = sorted(d for d in root.iterdir() if d.is_dir())
    sem_dir = Path(args.sem_dir) if args.sem_dir else \
        next(d for d in subs if "CDSEM" in d.name.upper())
    dom_dir = Path(args.dom_dir) if args.dom_dir else \
        next(d for d in subs if "PATTERN" in d.name.upper() or "DOM" in d.name.upper())
    print(f"SEM: {sem_dir.name}\nDOM: {dom_dir.name}\n")

    sem = imgs(sem_dir)
    dom = imgs(dom_dir)

    # ---------- 1. 檔名結構分析 ----------
    def family(stem):
        return "".join(sorted(k for k, _ in TOKEN.findall(stem.replace("_DOM", ""))))

    for name, files in (("SEM", sem), ("DOM", dom)):
        fam = Counter(family(p.stem) for p in files)
        print(f"{name} 檔名家族（token 字母組合 -> 數量）:")
        for k, v in fam.most_common():
            ex = next(p.stem for p in files if family(p.stem) == k)
            print(f"   {k:<12} {v:>5}   例: {ex}")
        print()

    # ---------- 2. 配對 ----------
    dom_by_key = defaultdict(list)
    for p in dom:
        dom_by_key[key_of(parse_tokens(p.stem))].append(p)
    dom_by_stem = {p.stem.replace("_DOM", ""): p for p in dom}

    pairs, unmatched = [], []
    for p in sem:
        q = dom_by_stem.get(p.stem)                      # 規則 A：僅差 _DOM
        if q is None:
            cand = dom_by_key.get(key_of(parse_tokens(p.stem)))  # 規則 B：token 順序無關
            q = cand[0] if cand else None
        (pairs.append({"sem": str(p), "dom": str(q), "stem": p.stem})
         if q else unmatched.append(p.stem))

    print(f"配對成功 {len(pairs)} / {len(sem)}，未配對 {len(unmatched)}")
    if unmatched:
        print(f"  未配對範例: {unmatched[:5]}")
        print("  -> 若集中在某個家族，代表 LW/SW 之間有換算關係（可能是 pitch = L + S），")
        print("     把上面的家族表貼給我，我補上換算規則。\n")

    # ---------- 3. 極性驗證（最關鍵）----------
    if pairs:
        print("=== 極性驗證 ===")
        same, inv = [], []
        for pr in pairs[:args.n_check]:
            s = np.array(Image.open(pr["sem"]).convert("L"), dtype=np.float32) / 255.0
            d = np.array(Image.open(pr["dom"]).convert("L"), dtype=np.float32) / 255.0
            m = d > 0.5
            if m.mean() in (0.0, 1.0):
                continue
            # DOM 白區 vs 黑區，在 SEM 上的平均灰階誰比較亮
            same.append(float(s[m].mean() - s[~m].mean()))
        arr = np.array(same)
        pos = int((arr > 0).sum())
        print(f"  取樣 {len(arr)} 張；DOM 白區在 SEM 上較亮的有 {pos} 張，較暗的有 {len(arr)-pos} 張")
        print(f"  平均亮度差 = {arr.mean():+.4f}")
        if arr.mean() < 0:
            print("  -> DOM 白區對應 SEM 暗區。**mask 需要反相**，")
            print("     或改以『DOM 黑區』為前景。請在 dataset 設 invert_mask=True。")
        else:
            print("  -> DOM 白區對應 SEM 亮區，極性一致，不需反相。")
        fg = np.mean([np.array(Image.open(p["dom"]).convert("L")).mean() / 255.0
                      for p in pairs[:args.n_check]])
        print(f"  DOM 白區平均佔比 = {fg:.3f}")
        print("  注意：前景佔比 >0.8 時，IoU 會被大面積的多數類別灌水，")
        print("        這也是必須報 Boundary IoU 的另一個理由。")

    Path(args.out).write_text(json.dumps(
        {"sem_dir": str(sem_dir), "dom_dir": str(dom_dir),
         "pairs": pairs, "unmatched": unmatched}, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
