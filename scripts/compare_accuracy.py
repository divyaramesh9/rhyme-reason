#!/usr/bin/env python3
"""Compare answer accuracy across runs (models) and conditions.

Reads one or more run directories (each must contain responses.jsonl +
run_config.json; scores are computed live with the existing scorer --
no trusting of stale scores.json), then prints, per run and dataset:

  * strict accuracy per condition (correct / attempted);
  * the accuracy difference vs each control condition present
    (answer_only, prose_reasoning, line_breaks);
  * parse failures, generation failures, and output-limit hits.

Every generated response counts -- compliant or not, correct or not.
Nothing is dropped, repaired, or inferred about poetic adherence.

Usage:
    python3 scripts/compare_accuracy.py \
        --run-dirs outputs/eval_qwen outputs/eval_llama outputs/eval_gpt4o
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import score_answers as sa
import study_defs

CONTROL_ORDER = ["answer_only", "prose_reasoning", "line_breaks"]


def compare(run_dirs: list[Path]) -> dict:
    out = {"runs": {}}
    for run_dir in run_dirs:
        if not (run_dir / "responses.jsonl").exists():
            out["runs"][str(run_dir)] = {"error": "no responses.jsonl"}
            continue
        report = sa.score_run(run_dir)
        cells = report["summary"]
        # regroup by dataset
        per_ds: dict[str, dict] = {}
        for cell, s in cells.items():
            dataset, cond = cell.split("/", 1)
            per_ds.setdefault(dataset, {})[cond] = {
                "attempted": s["attempted"],
                "correct": s["correct"],
                "accuracy_strict": s["accuracy_strict"],
                "parse_status": s["parse_status"],
                "generation_failures": s["generation_failures"],
                "hit_output_limit": s["hit_output_limit"],
            }
        deltas = {}
        for dataset, conds in per_ds.items():
            for cond, s in conds.items():
                for ctrl in CONTROL_ORDER:
                    if ctrl in conds and ctrl != cond:
                        base = conds[ctrl]["accuracy_strict"]
                        if base is not None and \
                                s["accuracy_strict"] is not None:
                            deltas[f"{dataset}:{cond}-vs-{ctrl}"] = round(
                                s["accuracy_strict"] - base, 4)
        out["runs"][str(run_dir)] = {
            "expected_pairs": report["expected_pairs"],
            "attempted_pairs": report["attempted_pairs"],
            "not_attempted_count": len(report["not_attempted"]),
            "per_dataset_condition": per_ds,
            "accuracy_deltas_vs_controls": deltas,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rep = compare(args.run_dirs)
    for run_dir, r in rep["runs"].items():
        print(f"\n=== {run_dir} ===")
        if "error" in r:
            print(f"  {r['error']}")
            continue
        if r["not_attempted_count"]:
            print(f"  WARNING: {r['not_attempted_count']} expected "
                  f"(example, condition) pairs not attempted -- "
                  f"INCOMPLETE RUN, not silently dropped")
        for dataset, conds in sorted(r["per_dataset_condition"].items()):
            print(f"  {dataset}:")
            for cond in sorted(conds):
                s = conds[cond]
                line = (f"    {cond:<20} {s['correct']}/{s['attempted']} "
                        f"(acc {s['accuracy_strict']})")
                if s["generation_failures"]:
                    line += f"  gen_failures={s['generation_failures']}"
                if s["hit_output_limit"]:
                    line += f"  hit_limit={s['hit_output_limit']}"
                bad_parse = {k: v for k, v in s["parse_status"].items()
                             if k != "parsed" and v}
                if bad_parse:
                    line += f"  parse={bad_parse}"
                print(line)
        if rep["runs"][str(run_dir)]["accuracy_deltas_vs_controls"]:
            print("  deltas vs controls:")
            for k, v in sorted(
                    rep["runs"][str(run_dir)]
                    ["accuracy_deltas_vs_controls"].items()):
                print(f"    {k}: {v:+.4f}")

    if args.out:
        args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False)
                            + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
