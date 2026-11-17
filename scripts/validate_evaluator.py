#!/usr/bin/env python3
"""Compare automated adherence verdicts with human screening labels.

Joins reviews/screening/review.csv (response-level labels) to the
deterministic / LLM-judge records in <run>/adherence_evals.jsonl via the
stable response_id. Rows whose recorded rule hash no longer matches the
current prompt file are flagged as INCOMPATIBLE and excluded from
agreement (never silently pooled).

Reports, overall and per device / model / dataset:
  * labeled rows, compatible rows, evaluated coverage;
  * confusion table over resolved pass/fail judgments;
  * agreement on resolved labels;
  * false-pass / false-fail counts (automated vs human);
  * uncertain / not_evaluable / failed / missing evaluations;
  * automated decision coverage.

Writes a disagreement CSV for inspection. Human labels are NEVER
modified -- automated predictions live only in adherence_evals.jsonl.

NOTE: the screening labels were used to DEVELOP this evaluator
(trailing-punctuation policy etc.). Agreement here measures
implementation fidelity, NOT independent validation. Independent
validation is the blinded fresh audit (scripts/audit_pack.py).

Usage:
    python3 scripts/validate_evaluator.py \
        --csv reviews/screening/review.csv \
        --run-dirs outputs/screening_qwen outputs/screening_llama \
            outputs/screening_gpt4o \
        --out reports/eval_validation.json \
        --disagreements reports/eval_disagreements.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import adherence as ad
import common
import eval_adherence as ea
import import_review as ir
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_FILE = ea.EVALS_FILE
RESOLVED = ("pass", "fail")


def automated_evals(run_dirs: list[Path]) -> dict[str, dict]:
    """response_id -> {source -> eval record}."""
    out: dict[str, dict] = {}
    for run_dir in run_dirs:
        for e in common.read_jsonl(run_dir / EVALS_FILE):
            if e["source"] in ("heuristic", "llm_judge"):
                out.setdefault(e["response_id"], {})[e["source"]] = e
    return out


def cell_stats() -> dict:
    return {
        "labeled": 0, "incompatible_rule": 0, "evaluated": 0,
        "resolved_pairs": 0, "agree": 0,
        "confusion": {f"h{h}_a{a}": 0 for h in RESOLVED for a in RESOLVED},
        "false_pass": 0, "false_fail": 0,
        "auto_uncertain": 0, "auto_not_evaluable": 0,
        "auto_processing_error": 0, "missing_eval": 0,
        "human_uncertain": 0, "human_not_evaluable": 0,
    }


def _update(stats: dict, human: str, verdict: str | None,
            status: str | None):
    if human == "uncertain":
        stats["human_uncertain"] += 1
    elif human == "not_evaluable":
        stats["human_not_evaluable"] += 1
    if verdict is None:
        stats["missing_eval"] += 1
        return
    stats["evaluated"] += 1
    if status not in (None, "resolved", "no_applicable_lines",
                      "missing_reasoning_view"):
        stats["auto_processing_error"] += 1
    if verdict not in RESOLVED:
        stats[f"auto_{verdict}"] = stats.get(f"auto_{verdict}", 0) + 1
        return
    if human not in RESOLVED:
        return
    stats["resolved_pairs"] += 1
    stats["confusion"][f"h{human}_a{verdict}"] += 1
    if human == verdict:
        stats["agree"] += 1
    elif verdict == "pass":
        stats["false_pass"] += 1
    else:
        stats["false_fail"] += 1


def _rate(stats: dict) -> dict:
    rp = stats["resolved_pairs"]
    return {**stats,
            "agreement_on_resolved": round(stats["agree"] / rp, 4)
            if rp else None,
            "coverage": round(stats["evaluated"] / stats["labeled"], 4)
            if stats["labeled"] else None}


def validate(csv_path: Path, run_dirs: list[Path],
             outputs_root: Path) -> dict:
    known = ir.known_response_ids(outputs_root)
    auto = automated_evals(run_dirs)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

    cur_sha = {c: ad.rule_sha256(c) for c in ad.DEVICE_EVALUATORS}
    cells = defaultdict(cell_stats)
    overall = cell_stats()
    disagreements = []
    n_unlabeled = 0

    for row in rows:
        rid = row.get("response_id", "")
        human = (row.get("adherence_label") or "").strip()
        if human not in study_defs.REVIEW_LABELS:
            n_unlabeled += 1
            continue
        info = known.get(rid)
        cond = (row.get("condition") or (info or {}).get("condition"))
        model = (row.get("model") or (info or {}).get("model") or "?")
        dataset = (row.get("dataset") or (info or {}).get("dataset") or "?")
        sources = auto.get(rid, {})
        ev = sources.get("llm_judge") or sources.get("heuristic")
        src = "llm_judge" if sources.get("llm_judge") else (
            "heuristic" if sources.get("heuristic") else None)

        key_ds = f"{cond}"
        key_md = f"{model} | {dataset} | {cond}"
        targets = [overall, cells[key_ds], cells[key_md]]
        for t in targets:
            t["labeled"] += 1
        # Rule-version compatibility: the frozen prompt must not have
        # drifted since the label was written.
        if (cond in cur_sha
                and row.get("device_rule_sha256")
                and row["device_rule_sha256"] != cur_sha[cond]):
            for t in targets:
                t["incompatible_rule"] += 1
            continue
        verdict = ev["verdict"] if ev else None
        status = ev.get("processing_status") if ev else None
        for t in targets:
            _update(t, human, verdict, status)
        if (verdict in RESOLVED and human in RESOLVED
                and verdict != human):
            disagreements.append({
                "response_id": rid, "model": model, "dataset": dataset,
                "condition": cond, "human": human, "auto": verdict,
                "auto_source": src,
                "evidence_summary": (ev or {}).get("evidence_summary", ""),
                "notes": row.get("notes", "")})

    def fin(d):
        return {k: _rate(v) for k, v in sorted(d.items())}

    return {
        "csv": str(csv_path),
        "note": ("Screening labels were used to develop this evaluator; "
                 "agreement measures implementation fidelity, not "
                 "independent validation. Use the blinded fresh audit "
                 "(audit_pack.py) for an unbiased estimate."),
        "labeled_rows": sum(1 for r in rows
                          if (r.get("adherence_label") or "").strip()
                          in study_defs.REVIEW_LABELS),
        "unlabeled_rows": n_unlabeled,
        "overall": _rate(overall),
        "by_device": fin({k: v for k, v in cells.items()
                          if " | " not in k}),
        "by_model_dataset_device": fin({k: v for k, v in cells.items()
                                        if " | " in k}),
        "disagreements": disagreements,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--outputs-root", type=Path,
                    default=REPO_ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--disagreements", type=Path, default=None)
    args = ap.parse_args()

    rep = validate(args.csv, args.run_dirs, args.outputs_root)
    o = rep["overall"]
    print(f"human-labeled rows: {rep['labeled_rows']} "
          f"({rep['unlabeled_rows']} unlabeled skipped)")
    print(f"OVERALL: evaluated {o['evaluated']}/{o['labeled']} "
          f"(coverage {o['coverage']}), resolved pairs "
          f"{o['resolved_pairs']}, agreement "
          f"{o['agreement_on_resolved']}, false_pass "
          f"{o['false_pass']}, false_fail {o['false_fail']}")
    print(f"  auto uncertain {o['auto_uncertain']}, not_evaluable "
          f"{o.get('auto_not_evaluable', 0)}, processing errors "
          f"{o['auto_processing_error']}, missing evals "
          f"{o['missing_eval']}, incompatible-rule rows "
          f"{o['incompatible_rule']}")
    for dev, s in rep["by_device"].items():
        print(f"  {dev:<14} labeled {s['labeled']:<3} resolved "
              f"{s['resolved_pairs']:<3} agree {s['agree']:<3} "
              f"fp {s['false_pass']} ff {s['false_fail']} "
              f"unc {s['auto_uncertain']} ne "
              f"{s.get('auto_not_evaluable', 0)} miss "
              f"{s['missing_eval']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2,
                                       ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")
    if args.disagreements:
        args.disagreements.parent.mkdir(parents=True, exist_ok=True)
        with open(args.disagreements, "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "response_id", "model", "dataset", "condition",
                "human", "auto", "auto_source", "evidence_summary",
                "notes"])
            w.writeheader()
            w.writerows(rep["disagreements"])
        print(f"wrote {len(rep['disagreements'])} disagreement(s) -> "
              f"{args.disagreements}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
