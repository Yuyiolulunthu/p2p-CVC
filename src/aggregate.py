

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

COLS = [("iou", "IoU", 4), ("biou_d2", "BIoU$_{d=2}$", 4),
        ("dice", "Dice", 4), ("epe_mean", "EPE", 2),
        ("cd_mae", "CD MAE", 2), ("fp_blobs", "FP blobs", 1),
        ("border_iou", "Border IoU", 4)]


def load(runs: Path):
    out = []
    for f in sorted(runs.glob("*/final_metrics.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        c = d["config"]
        out.append({"name": f.parent.name, "cfg": c, "test": d["test"],
                    "val": d["val"], "enc_M": d["encoder_params_M"],
                    "tot_M": d["total_params_M"]})
    return out


def agg(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / max(n - 1, 1)) if n > 1 else 0.0
    return m, s, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper/table.tex")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--loco", action="store_true", help="改為 per-class 呈現")
    args = ap.parse_args()

    rows = load(Path(args.runs))
    if not rows:
        raise SystemExit(f"no final_metrics.json under {args.runs}")

    groups = defaultdict(list)
    for r in rows:
        c = r["cfg"]
        key = (c.get("loco_class") or "-") if args.loco else \
            (c["encoder"], c.get("variant", "v0"), c.get("lambda_tv", 0))
        groups[key].append(r)

    head = ["Method" if not args.loco else "Held-out class", "\\#Enc (M)"] + \
           [c[1] for c in COLS]
    lines = ["\\begin{tabular}{l" + "c" * (len(head) - 1) + "}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]

    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        label = key if isinstance(key, str) else \
            f"{key[0]} / {key[1]}" + (f" ($\\lambda_{{tv}}$={key[2]})" if key[2] else "")
        cells = [label.replace("_", "\\_"), f"{rs[0]['enc_M']:.2f}"]
        for k, _, p in COLS:
            m, s, n = agg([r[args.split][k] for r in rs])
            cells.append(f"{m:.{p}f}" + (f" $\\pm$ {s:.{p}f}" if n > 1 else ""))
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
