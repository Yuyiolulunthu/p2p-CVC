"""
把 runs/ 底下所有 final_metrics.json 收成表。

    python -m src.aggregate                        # 掃 runs/，印 seed 平均表
    python -m src.aggregate --runs runs --latex paper/table1.tex
    python -m src.aggregate --per-class            # 加印最佳模型的 per-class 細項
    python -m src.aggregate --loco                 # 只看 LOCO（每個 class 一列）

分組規則：run 名稱結尾的 `_s<digit>` 視為 seed，去掉之後相同的歸為一組。
    resnet34_s0 / resnet34_s1 / resnet34_s2  ->  resnet34，n=3

為什麼要有這支：單一 seed 的排名在這份資料上翻過兩次。只報一個數字的表，
讀者沒辦法判斷差距是真的還是雜訊，所以每一欄都要帶 std。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# 要報的欄位：(json key, 顯示名, 小數位, 越小越好?)
COLS = [
    ("iou",         "IoU",      4, False),
    ("biou_d2",     "BIoU",     4, False),
    ("epe_mean",    "EPE",      3, True),
    ("epe_pred2gt", "EPE p2g",  3, True),
    ("cd_mae",      "CD MAE",   2, True),
    ("cd_bias",     "CD bias",  2, None),    # None = 越接近 0 越好
    ("fp_blobs",    "FP blob",  1, True),
]

# 論文用的正式名稱
PRETTY = {
    "resnet34":                   "ResNet-34",
    "mobileone_s1":               "MobileOne-S1",
    "mobileone_s0":               "MobileOne-S0",
    "mit_b0":                     "SegFormer (MiT-B0)",
    "efficientnetb0":             "EfficientNet-B0",
    "efficientnet-b0":            "EfficientNet-B0",
    "timmmobilenetv3_large_100":  "MobileNetV3-Large",
    "timm-mobilenetv3_large_100": "MobileNetV3-Large",
}

SEED_RE = re.compile(r"_s(\d+)$")


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)      # 樣本標準差
    return m, math.sqrt(var)


def load_runs(root: Path) -> list[dict]:
    out = []
    for f in sorted(root.glob("*/final_metrics.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                                  # noqa: BLE001
            print(f"  [skip] {f.parent.name}: {type(e).__name__}")
            continue
        if "test" not in d:
            print(f"  [skip] {f.parent.name}: 沒有 test 區塊（可能是被中斷的 run）")
            continue
        d["_name"] = f.parent.name
        out.append(d)
    return out


def group_key(name: str, cfg: dict) -> tuple[str, str]:
    """回傳 (組名, 種類)。種類是 'sweep' 或 'loco'。"""
    held = cfg.get("held_out") or cfg.get("class") or cfg.get("held_out_class")
    if held:
        # LOCO 要連 backbone 一起分組，否則不同 encoder 的同一個留出 class
        # 會被平均成一列，那是把兩個實驗混在一起。
        enc = cfg.get("encoder", "?")
        return f"{PRETTY.get(enc, enc)} / {held}", "loco"
    if name.lower().startswith(("loco_", "lofo_")):
        return name.split("_", 1)[1], "loco"
    return SEED_RE.sub("", name), "sweep"


def fmt(m: float, s: float, nd: int) -> str:
    if math.isnan(s):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}±{s:.{nd}f}"


def best_index(rows: list[tuple], col_i: int, lower_better) -> int:
    """rows[i][2] 是 {key: (mean,std)}；回傳該欄最佳那一列的 index。"""
    key = COLS[col_i][0]
    vals = [r[2][key][0] for r in rows]
    if lower_better is None:
        vals = [abs(v) for v in vals]
        lower_better = True
    return (min if lower_better else max)(range(len(vals)), key=lambda i: vals[i])


def print_table(title: str, rows: list[tuple], extra_hdr: str = "params"):
    print(f"\n=== {title} ===")
    w = max(len(r[0]) for r in rows) + 2
    hdr = f"{'model'.ljust(w)}{'n':>3} {extra_hdr:>9}"
    for _, disp, _, _ in COLS:
        hdr += f" {disp:>13}"
    print(hdr)
    print("-" * len(hdr))

    marks = {}
    for i, (_, _, _, lb) in enumerate(COLS):
        marks[best_index(rows, i, lb)] = marks.get(best_index(rows, i, lb), 0)

    for ri, (name, n, agg, extra) in enumerate(rows):
        line = f"{name.ljust(w)}{n:>3} {extra:>9}"
        for ci, (key, _, nd, lb) in enumerate(COLS):
            m, s = agg[key]
            cell = fmt(m, s, nd)
            star = "*" if ri == best_index(rows, ci, lb) else " "
            line += f" {cell:>12}{star}"
        print(line)
    print("  * = 該欄最佳（CD bias 以絕對值最小為佳）")


def latex_table(rows: list[tuple], path: Path, caption: str, label: str):
    cols = "l r r " + " ".join("r" for _ in COLS)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        "Encoder & Params (M) & $n$ & "
        + " & ".join(d for _, d, _, _ in COLS) + r" \\",
        r"\midrule",
    ]
    for ri, (name, n, agg, extra) in enumerate(rows):
        cells = []
        for ci, (key, _, nd, lb) in enumerate(COLS):
            m, s = agg[key]
            c = f"{m:.{nd}f}" if math.isnan(s) else rf"{m:.{nd}f}\,$\pm$\,{s:.{nd}f}"
            if ri == best_index(rows, ci, lb):
                c = rf"\textbf{{{c}}}"
            cells.append(c)
        lines.append(f"{name} & {extra} & {n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nLaTeX -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--latex", default=None, help="輸出 LaTeX 表格的路徑")
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--loco", action="store_true", help="只看 LOCO 的 run")
    ap.add_argument("--raw", action="store_true", help="逐一列出每個 run，不做平均")
    args = ap.parse_args()

    root = Path(args.runs)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"{root} 底下找不到任何 final_metrics.json")

    groups: dict[str, list[dict]] = defaultdict(list)
    kinds: dict[str, str] = {}
    for d in runs:
        g, kind = group_key(d["_name"], d.get("config", {}))
        if args.loco and kind != "loco":
            continue
        if not args.loco and kind == "loco":
            continue
        groups[g].append(d)
        kinds[g] = kind

    if not groups:
        raise SystemExit("符合條件的 run 是空的（--loco 開了但沒有 LOCO run？）")

    if args.raw:
        print(f"\n=== 每個 run（{args.split}）===")
        for d in sorted(runs, key=lambda x: x["_name"]):
            t = d[args.split]
            cells = " ".join(f"{disp}={t[k]:.{nd}f}" for k, disp, nd, _ in COLS)
            print(f"  {d['_name']:<30} ep{d.get('best_epoch','?'):>4}  {cells}")

    rows = []
    for g in sorted(groups):
        ds = groups[g]
        agg = {k: mean_std([d[args.split][k] for d in ds]) for k, _, _, _ in COLS}
        p = ds[0].get("total_params_M")
        extra = f"{p:.2f}" if isinstance(p, (int, float)) else "–"
        rows.append((PRETTY.get(g, g), len(ds), agg, extra))

    # 掃描表按參數量由小到大排，正好對應「換小 encoder」這個主題；
    # LOCO 按 class 名排。
    if not args.loco:
        rows.sort(key=lambda r: float(r[3]) if r[3] != "\u2013" else 1e9)

    title = ("Leave-one-class-out" if args.loco
             else f"Encoder sweep ({args.split})")
    print_table(title, rows, extra_hdr="params")

    if args.latex:
        latex_table(
            rows, Path(args.latex),
            caption=("Leave-one-class-out generalization. Each row trains on 11 "
                     "pattern classes and tests on the held-out class."
                     if args.loco else
                     "Encoder comparison on the random split. Mean $\\pm$ sample "
                     "standard deviation over three seeds."),
            label=("tab:loco" if args.loco else "tab:encoders"),
        )

    # ---- per-class 細項 ----
    if args.per_class:
        pc_key = "test_per_class"
        have = [d for d in runs if pc_key in d]
        if not have:
            print(f"\n（沒有 {pc_key}，跳過 per-class）")
            return
        # 以 BIoU 最佳的組當代表
        best_g = max(groups, key=lambda g: sum(
            d[args.split]["biou_d2"] for d in groups[g]) / len(groups[g]))
        ds = groups[best_g]
        classes = sorted(ds[0][pc_key])
        print(f"\n=== per-class — {PRETTY.get(best_g, best_g)}（{len(ds)} seeds 平均）===")
        print(f"{'class':<26}{'n':>4}{'IoU':>16}{'BIoU':>16}{'EPE':>14}")
        print("-" * 76)
        for c in classes:
            n = ds[0][pc_key][c].get("n", "?")
            cells = []
            for k, nd in (("iou", 4), ("biou_d2", 4), ("epe_mean", 3)):
                m, s = mean_std([d[pc_key][c][k] for d in ds])
                cells.append(fmt(m, s, nd))
            print(f"{c:<26}{n:>4}{cells[0]:>16}{cells[1]:>16}{cells[2]:>14}")
        print("\n注意：各 class 的前景佔比從 0.10 到 0.89，IoU 欄不可橫向比較；"
              "跨 class 的比較看 BIoU 與 EPE。")


if __name__ == "__main__":
    main()