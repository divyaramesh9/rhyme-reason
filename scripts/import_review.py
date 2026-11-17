#!/usr/bin/env python3
"""Import a completed review CSV, validate it, and summarize labels.

Reads the CSV written by prepare_review.py, verifies every row against
the saved responses under --outputs-root (default outputs/), checks
labels against the allowed set (pass | fail | uncertain |
not_evaluable | empty), then prints label counts by
model x dataset x condition, including unanswered reviews and
uncertain/not_evaluable cases explicitly.

Nothing is inferred: empty labels stay unlabeled, unknown response_ids
and duplicates are reported as errors, and correctness is never used.

Usage:
    python3 scripts/import_review.py --csv reviews/screening_qwen/review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import common
import prepare_review as pr
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent


def known_response_ids(outputs_root: Path) -> dict[str, dict]:
    """response_id -> {run_dir, dataset, condition, model} from every
    run under outputs_root."""
    known = {}
    for resp_path in sorted(outputs_root.glob("*/responses.jsonl")):
        run_dir = resp_path.parent
        cfg_path = run_dir / "run_config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        compat = cfg.get("compatibility", {})
        run_id = cfg.get("run_name", run_dir.name)
        fallback_model = {
            "repo": compat.get("model_repo"),
            "revision": compat.get("model_revision"),
        }
        for rec in common.read_jsonl(resp_path):
            model = rec.get("model") or fallback_model
            rid = pr.response_id(run_id, model, compat, rec)
            known[rid] = {
                "run_id": run_id,
                "dataset": rec["dataset"],
                "condition": rec["condition"],
                "model": f"{model.get('repo')}@{model.get('revision')}",
            }
    return known


def import_review(csv_path: Path, outputs_root: Path) -> dict:
    known = known_response_ids(outputs_root)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    errors = []
    seen = set()
    for i, row in enumerate(rows, 2):  # header = line 1
        rid = row.get("response_id", "")
        label = (row.get("adherence_label") or "").strip()
        if not rid:
            errors.append(f"line {i}: empty response_id")
            continue
        if rid in seen:
            errors.append(f"line {i}: duplicate response_id {rid}")
        seen.add(rid)
        if rid not in known:
            errors.append(f"line {i}: unknown response_id {rid}")
        if label and label not in study_defs.REVIEW_LABELS:
            errors.append(
                f"line {i}: invalid label {label!r} (allowed: "
                f"{study_defs.REVIEW_LABELS} or empty)")

    cells = defaultdict(lambda: {
        "total": 0, "labeled": 0,
        **{l: 0 for l in study_defs.REVIEW_LABELS},
        "unlabeled": 0,
    })
    for row in rows:
        rid = row.get("response_id", "")
        info = known.get(rid)
        if info is None:
            continue
        key = (info["model"], info["dataset"], info["condition"])
        label = (row.get("adherence_label") or "").strip()
        cells[key]["total"] += 1
        if label in study_defs.REVIEW_LABELS:
            cells[key][label] += 1
            cells[key]["labeled"] += 1
        else:
            cells[key]["unlabeled"] += 1

    summary = {
        f"{m} | {d} | {c}": v
        for (m, d, c), v in sorted(cells.items())
    }
    return {
        "csv": str(csv_path),
        "rows": len(rows),
        "errors": errors,
        "summary": summary,
        "labels_used": sorted({
            (r.get("adherence_label") or "").strip()
            for r in rows if (r.get("adherence_label") or "").strip()
        }),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--outputs-root", type=Path,
                    default=REPO_ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON summary path (default <csv>.summary.json)")
    args = ap.parse_args()

    report = import_review(args.csv, args.outputs_root)
    out = args.out or args.csv.with_suffix(".summary.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False)
                   + "\n", encoding="utf-8")

    print(f"imported {report['rows']} review row(s) from {args.csv}")
    for cell, v in report["summary"].items():
        print(f"  {cell}: labeled {v['labeled']}/{v['total']} "
              f"(pass {v['pass']}, fail {v['fail']}, uncertain "
              f"{v['uncertain']}, not_evaluable {v['not_evaluable']}, "
              f"unlabeled {v['unlabeled']})")
    n_unlabeled = sum(v["unlabeled"] for v in report["summary"].values())
    if n_unlabeled:
        print(f"  {n_unlabeled} review(s) still unanswered")
    if report["errors"]:
        print(f"  {len(report['errors'])} problem row(s):",
              file=sys.stderr)
        for e in report["errors"]:
            print(f"    {e}", file=sys.stderr)
    print(f"wrote {out}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
