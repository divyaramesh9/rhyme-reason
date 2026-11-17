#!/usr/bin/env python3
"""Blinded fresh human-audit workflow for the pilot evaluator.

prepare   -- draw a seeded random subset of pilot responses: one
             response per (model x dataset x device) cell (30 cells at
             the current 3 x 2 x 5 configuration). IDs are selected and
             recorded BEFORE any automatic verdict is consulted; the
             review files contain the device rule and the unaltered raw
             response only -- never the automatic verdict or answer
             correctness.
import    -- merge your labels into the runs' adherence_evals.jsonl as
             source "human_audit" (human labels are never altered; the
             CSV is re-validated on every import).
decide    -- record your per-device acceptance decision for the
             automated evaluator (accepted | provisional | rejected).
summary   -- compare imported audit labels with automated verdicts on
             the sampled IDs only.

The audit subset is deliberately small (one per cell). A
disagreement-enriched review is a separate workflow -- it cannot
estimate overall agreement and is never pooled into audit stats.

Usage:
    python3 scripts/audit_pack.py prepare \
        --run-dirs outputs/dev_pilot_qwen outputs/dev_pilot_llama \
            outputs/dev_pilot_gpt4o --out-dir reviews/pilot_audit
    python3 scripts/audit_pack.py import --csv reviews/pilot_audit/audit.csv
    python3 scripts/audit_pack.py summary --out-dir reviews/pilot_audit
    python3 scripts/audit_pack.py decide --out-dir reviews/pilot_audit \
        --device meter --decision provisional --note "..."
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import adherence as ad
import common
import eval_adherence as ea
import prepare_review as pr
import prompt_lib
import response_parsing as rp
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_FILE = ea.EVALS_FILE

CSV_FIELDS = ["audit_id", "dataset", "example_id", "condition",
              "device_rule_sha256", "response_file",
              "adherence_label", "notes"]


def _cells(run_dirs: list[Path]):
    """(model_repo, dataset, device) -> [(response_id, rec, meta)]."""
    cells: dict[tuple, list] = {}
    for run_dir in run_dirs:
        for rid, rec, meta in ea.iter_evaluatable(run_dir):
            model = (rec.get("model") or meta["fallback_model"])
            key = (model.get("repo"), rec["dataset"], rec["condition"])
            cells.setdefault(key, []).append((rid, rec, meta, run_dir))
    return cells


def cmd_prepare(args) -> int:
    cells = _cells(args.run_dirs)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # blinding.json is the UNBLINDING KEY: it links each audit_id back to
    # its response_id/run/model. Do NOT open it while annotating.
    blinding = {
        "WARNING": ("UNBLINDING KEY -- do not open while annotating. "
                    "IDs were selected before automatic verdicts were "
                    "consulted; review files and audit.csv contain no "
                    "model identity, automatic verdict, or correctness."),
        "seed": args.seed,
        "sample_per_cell": 1,
        "cells": {},
        "audit_ids": {},
    }
    rows = []
    audit_n = 0
    for (repo, dataset, device), cands in sorted(cells.items()):
        cands = sorted(cands, key=lambda c: c[0])
        rid, rec, meta, run_dir = rng.choice(cands)
        model = rec.get("model") or meta["fallback_model"]
        rule_sha = ad.rule_sha256(rec["condition"])
        rule = (ad.PROMPTS_DIR / f"{rec['condition']}.txt").read_text(
            encoding="utf-8")
        raw = rp.record_raw_text(rec)
        audit_n += 1
        audit_id = f"audit-{audit_n:03d}"
        dest = (args.out_dir / "files" / dataset / device
                / f"{audit_id}.audit.txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join([
            "BLINDED AUDIT -- judge ONLY adherence to the device rule.",
            "No model identity, automatic judgment, or correctness "
            "result is shown.",
            "",
            f"audit_id: {audit_id}",
            f"dataset: {dataset}  example_id: {rec['example_id']}  "
            f"condition: {device}",
            "",
            "=== DEVICE INSTRUCTIONS (verbatim) ===",
            rule,
            "",
            "=== RAW RESPONSE (unaltered) ===",
            raw,
            "",
        ]), encoding="utf-8")
        blinding["cells"][f"{repo} | {dataset} | {device}"] = {
            "candidates": len(cands), "selected_audit_id": audit_id}
        blinding["audit_ids"][audit_id] = {
            "response_id": rid, "run_id": meta["run_id"],
            "run_dir": str(run_dir),
            "model": f"{repo}@{model.get('revision')}",
            "dataset": dataset, "example_id": rec["example_id"],
            "condition": device}
        rows.append({
            "audit_id": audit_id,
            "dataset": dataset, "example_id": rec["example_id"],
            "condition": device, "device_rule_sha256": rule_sha,
            "response_file": str(dest),
            "adherence_label": "", "notes": ""})

    (args.out_dir / "blinding.json").write_text(
        json.dumps(blinding, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with open(args.out_dir / "audit.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"cells: {len(cells)} -> {len(rows)} blinded audit file(s) "
          f"(seed {args.seed})")
    print(f"wrote {args.out_dir / 'audit.csv'} -- fill "
          "adherence_label with pass|fail|uncertain|not_evaluable, "
          "then: audit_pack.py import")
    print(f"blinding key kept in {args.out_dir / 'blinding.json'} -- "
          "do not open until annotation is complete")
    return 0


def cmd_import(args) -> int:
    out_dir = args.out_dir or args.csv.parent
    blinding_path = out_dir / "blinding.json"
    if not blinding_path.exists():
        print(f"error: {blinding_path} not found -- the unblinding key "
              "written by 'prepare' is required to link labels back to "
              "responses", file=sys.stderr)
        return 1
    key = json.loads(blinding_path.read_text(encoding="utf-8"))[
        "audit_ids"]
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    errors, n = [], 0
    for i, row in enumerate(rows, 2):
        label = (row.get("adherence_label") or "").strip()
        if not label:
            continue
        if label not in study_defs.REVIEW_LABELS:
            errors.append(f"line {i}: invalid label {label!r}")
            continue
        info = key.get(row.get("audit_id", ""))
        if info is None:
            errors.append(f"line {i}: unknown audit_id "
                          f"{row.get('audit_id')!r}")
            continue
        rid = info["response_id"]
        run_dir = Path(info["run_dir"])
        evals_path = run_dir / EVALS_FILE
        if not run_dir.exists():
            errors.append(f"line {i}: run dir {run_dir} not found")
            continue
        ekey = common.sha256_text(
            "|".join([rid, "human_audit", row.get(
                "device_rule_sha256", "")]))
        done = {r["eval_key"] for r in common.read_jsonl(evals_path)}
        if ekey in done:
            continue
        common.append_jsonl(evals_path, {
            "eval_key": ekey, "response_id": rid,
            "run_id": info["run_id"], "dataset": info["dataset"],
            "example_id": info["example_id"],
            "condition": info["condition"], "source": "human_audit",
            "verdict": label, "processing_status": "resolved",
            "units": [], "evidence_summary": row.get("notes", ""),
            "rule_sha256": row.get("device_rule_sha256"),
            "audit_id": row["audit_id"],
            "notes": row.get("notes", ""),
            "audit_csv": str(args.csv),
            "recorded_at": common.utc_now()})
        n += 1
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"imported {n} human-audit label(s)")
    return 1 if errors else 0


def cmd_decide(args) -> int:
    path = args.out_dir / "evaluator_acceptance.json"
    data = json.loads(path.read_text(encoding="utf-8")) \
        if path.exists() else {"devices": {}}
    data["devices"][args.device] = {
        "decision": args.decision, "note": args.note,
        "recorded_at": common.utc_now()}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"{args.device}: {args.decision} -> {path}")
    print("decisions are recorded, not enforced: provisional devices keep "
          "their automated results labeled provisional in reports.")
    return 0


def cmd_summary(args) -> int:
    sel_path = args.out_dir / "blinding.json"
    csv_path = args.out_dir / "audit.csv"
    if not sel_path.exists() or not csv_path.exists():
        print("error: run 'prepare' first (and fill + 'import' the CSV)",
              file=sys.stderr)
        return 1
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    aid2rid = {a: v["response_id"]
               for a, v in sel["audit_ids"].items()}
    labels = {r["audit_id"]: (r.get("adherence_label") or "").strip()
              for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    auto = {}
    for run_dir in args.run_dirs:
        for e in common.read_jsonl(run_dir / EVALS_FILE):
            if e["source"] in ("heuristic", "llm_judge"):
                auto.setdefault(e["response_id"], {})[e["source"]] = e
    rows = []
    for aid in sorted(aid2rid):
        rid = aid2rid[aid]
        human = labels.get(aid) or "(unlabeled)"
        ev = (auto.get(rid, {}).get("llm_judge")
              or auto.get(rid, {}).get("heuristic"))
        verdict = ev["verdict"] if ev else "(not evaluated)"
        rows.append({"response_id": rid, "human": human,
                     "auto": verdict,
                     "agree": human == verdict})
        print(f"  {verdict:<14} auto vs human {human:<14} "
              f"{aid} ({rid})")
    res = [r for r in rows if r["human"] in ("pass", "fail")
           and r["auto"] in ("pass", "fail")]
    agree = sum(r["agree"] for r in res)
    print(f"\nresolved pass/fail pairs: {len(res)}, agree {agree} "
          f"({round(agree / len(res), 3) if res else 'n/a'})")
    print("CAVEAT: one response per cell -- a small, indicative sample, "
          "not a reliability estimate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261116)
    p.set_defaults(fn=cmd_prepare)

    i = sub.add_parser("import")
    i.add_argument("--csv", type=Path, required=True)
    i.add_argument("--out-dir", type=Path, default=None)
    i.add_argument("--outputs-root", type=Path,
                   default=REPO_ROOT / "outputs")
    i.set_defaults(fn=cmd_import)

    s = sub.add_parser("summary")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    s.set_defaults(fn=cmd_summary)

    d = sub.add_parser("decide")
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--device", required=True,
                   choices=list(ad.DEVICE_EVALUATORS))
    d.add_argument("--decision", required=True,
                   choices=["accepted", "provisional", "rejected"])
    d.add_argument("--note", default="")
    d.set_defaults(fn=cmd_decide)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
