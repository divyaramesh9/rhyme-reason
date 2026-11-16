#!/usr/bin/env python3
"""Plot a manual-adherence review summary as a grouped stacked bar chart.

Reads a review summary JSON produced by import_review.py and writes one
PNG per invocation: for each dataset, one subplot where the x-axis is the
condition/device, each device has one bar per model, and each bar is
stacked by label.  Model is encoded by bar color; label is encoded by
fill style (pass = solid, uncertain = pale, fail = striped,
not_evaluable = dotted, unlabeled = empty outline).

Requires matplotlib (install separately; not in requirements.txt):

    python3 scripts/plot_review.py \
        --summary reviews/screening/review.summary.json \
        --out reviews/screening/review_plot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import study_defs

MODEL_COLORS = ["#1f78b4", "#33a02c", "#6a3d9a", "#b15928"]
LABEL_ORDER = ["pass", "fail", "uncertain", "not_evaluable", "unlabeled"]
LABEL_STYLE = {
    "pass":          {"alpha": 1.0,  "hatch": None},
    "fail":          {"alpha": 0.85, "hatch": "//"},
    "uncertain":     {"alpha": 0.40, "hatch": None},
    "not_evaluable": {"alpha": 0.45, "hatch": "xx"},
    "unlabeled":     {"alpha": 0.12, "hatch": None},
}
LABEL_LABEL = {
    "pass": "pass (solid)",
    "fail": "fail (striped)",
    "uncertain": "uncertain (pale)",
    "not_evaluable": "not_evaluable (crossed)",
    "unlabeled": "unlabeled (empty)",
}


def short_model(name: str) -> str:
    """'mlx-community/Qwen2.5-7B-Instruct-4bit@c26a38f6a37d...' ->
    'Qwen2.5-7B-Instruct-4bit'; 'openai@gpt-4o-2024-08-06' -> 'gpt-4o-2024-08-06'."""
    if "@" in name and "/" in name:
        repo = name.split("@", 1)[0]
        return repo.split("/")[-1]
    return name.split("@", 1)[-1] if "@" in name else name


def parse_cells(summary: dict) -> dict:
    """cell key 'model | dataset | condition' -> nested dict."""
    cells = {}
    for key, v in summary["summary"].items():
        model, dataset, cond = [p.strip() for p in key.split("|")]
        cells.setdefault(dataset, {}).setdefault(model, {})[cond] = v
    return cells


def plot(summary: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        raise SystemExit(
            "error: matplotlib is not installed in this interpreter. "
            "Install it (e.g. 'python3 -m pip install matplotlib') or run "
            "with the python that already has it.")

    cells = parse_cells(summary)
    datasets = sorted(cells)
    models = sorted({m for d in cells.values() for m in d})
    conds = sorted({c for d in cells.values() for m in d.values()
                    for c in m})
    n_ds, n_m, n_c = len(datasets), len(models), len(conds)

    fig, axes = plt.subplots(
        n_ds, 1, figsize=(max(11, n_c * (n_m + 1) * 1.0), 4.8 * n_ds),
        squeeze=False)
    bar_w = 0.8 / n_m

    for ax, dataset in zip(axes.flat, datasets):
        x = list(range(n_c))
        for mi, model in enumerate(models):
            bottoms = [0.0] * n_c
            for label in LABEL_ORDER:
                vals = [cells[dataset].get(model, {})
                        .get(cond, {}).get(label, 0) for cond in conds]
                style = LABEL_STYLE[label]
                pos = [xi + mi * bar_w for xi in x]
                ax.bar(pos, vals, bar_w * 0.9, bottom=bottoms,
                       color=MODEL_COLORS[mi % len(MODEL_COLORS)],
                       alpha=style["alpha"], hatch=style["hatch"],
                       edgecolor="black", linewidth=0.4)
                bottoms = [b + v for b, v in zip(bottoms, vals)]
        center = [xi + (n_m - 1) * bar_w / 2 for xi in x]
        ax.set_xticks(center)
        ax.set_xticklabels(conds, rotation=25, ha="right")
        ax.set_ylabel("responses")
        ax.set_title(f"{dataset} -- manual adherence labels per device")
        ymax = max(
            (sum(v.get(l, 0) for l in LABEL_ORDER)
             for m in cells[dataset].values() for v in m.values()),
            default=1)
        ax.set_ylim(0, ymax + 0.6)
        ax.set_yticks(range(ymax + 1))
        ax.grid(axis="y", alpha=0.3)

    handles = [Patch(facecolor=MODEL_COLORS[i % len(MODEL_COLORS)],
                     edgecolor="black", label=short_model(m))
               for i, m in enumerate(models)]
    handles += [Patch(facecolor="#bbbbbb", edgecolor="black",
                      alpha=LABEL_STYLE[l]["alpha"],
                      hatch=LABEL_STYLE[l]["hatch"],
                      label=LABEL_LABEL[l]) for l in LABEL_ORDER]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5), fontsize=9,
               title="bar color = model    |    fill style = label")
    fig.suptitle("Adherence review: labels per model x dataset x device")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default <summary with .plot.png suffix>")
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    out = args.out or args.summary.with_suffix(".plot.png")
    plot(summary, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
