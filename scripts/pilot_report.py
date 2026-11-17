#!/usr/bin/env python3
"""Assemble the midterm-ready Markdown report for the development pilot.

Reads only saved records -- generation responses, derived adherence
evals, human-audit imports, the budget ledger -- never generates,
judges, or fabricates numbers. Missing data is reported as missing.

Sections: frozen configuration + provenance, coverage, accuracy,
matched accuracy differences vs the three controls, evaluator-assessed
adherence + joint success, validation vs screening labels, fresh-audit
results, generation vs judging costs, limitations.

Usage:
    python3 scripts/pilot_report.py \
        --run-dirs outputs/dev_pilot_qwen outputs/dev_pilot_llama \
            outputs/dev_pilot_gpt4o \
        --out reports/dev_pilot_midterm.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import adherence as ad
import budget as budget_mod
import build_dev_pilot as bdp
import common
import eval_adherence as ea
import score_answers as sa
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUDGET_CONFIG = REPO_ROOT / "configs" / "budget.json"
CONTROL_ORDER = ["answer_only", "prose_reasoning", "line_breaks"]

AUTO_SOURCE_ORDER = ["adjudicated", "llm_judge", "heuristic"]


def auto_verdict(evals_by_rid: dict, rid: str) -> dict | None:
    """The automated verdict for a response: prefer judge > heuristic.
    Human sources are never folded into automated estimates."""
    srcs = evals_by_rid.get(rid, {})
    for s in AUTO_SOURCE_ORDER:
        if srcs.get(s) and srcs[s].get("verdict") in ad.VERDICTS:
            return srcs[s]
    return None


def collect_run(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "run_config.json").read_text(
        encoding="utf-8"))
    compat = cfg.get("compatibility", {})
    scored = sa.score_run(run_dir)
    evals = common.read_jsonl(run_dir / ea.EVALS_FILE)
    by_rid: dict[str, dict] = {}
    for e in evals:
        by_rid.setdefault(e["response_id"], {})[e["source"]] = e
    return {"dir": run_dir, "config": cfg, "compat": compat,
            "scored": scored, "evals": evals, "by_rid": by_rid,
            "model": compat.get("model_repo", "?")}


def matched_delta(per_response: list[dict], dataset: str,
                  cond: str, ctrl: str) -> dict:
    """Accuracy difference cond - ctrl over question IDs present in
    BOTH conditions (matched comparison; incompleteness flagged)."""
    correct = {(r["example_id"], r["condition"]): r["correct"]
               for r in per_response if r["dataset"] == dataset}
    ids = ({e for (e, c) in correct if c == cond}
           & {e for (e, c) in correct if c == ctrl})
    n_cond = sum(1 for e in ids if correct.get((e, cond)))
    n_ctrl = sum(1 for e in ids if correct.get((e, ctrl)))
    total_cond = sum(1 for (e, c) in correct if c == cond)
    total_ctrl = sum(1 for (e, c) in correct if c == ctrl)
    acc_c = n_cond / len(ids) if ids else None
    acc_b = n_ctrl / len(ids) if ids else None
    return {
        "matched_ids": len(ids),
        "cond_total": total_cond, "ctrl_total": total_ctrl,
        "delta_pp": round((acc_c - acc_b) * 100, 2)
        if acc_c is not None and acc_b is not None else None,
        "incomplete": len(ids) < min(total_cond, total_ctrl),
    }


def report(run_dirs: list[Path], pilot_config: Path | None,
           audit_dir: Path | None, out: Path) -> str:
    lines = []
    w = lines.append
    w("# Development pilot -- midterm report\n")
    w("Automated adherence figures are **evaluator-assessed**, not "
      "verified ground truth. Screening (8 devices) and the frozen "
      "pilot (5 devices) are separate rounds.\n")

    # ---- configuration & provenance ----------------------------------
    w("## Configuration and provenance\n")
    if pilot_config and pilot_config.exists():
        cfg = json.loads(pilot_config.read_text(encoding="utf-8"))
        w(f"- pilot config: `{pilot_config}` "
          f"(questions_per_dataset={cfg['questions_per_dataset']}, "
          f"seed={cfg['seed']})")
        w(f"- devices (frozen): {', '.join(cfg['devices'])}")
        w(f"- controls: {', '.join(cfg['controls'])}")
        w("- anaphora/epistrophe use the **supplied** phrases "
          "('We know ' / 'as we see'); alliteration, rhyme_scheme and "
          "meter require additional sound/rhythm choices by the model.")
        frozen = REPO_ROOT / "configs" / "device_selection.frozen.json"
        if frozen.exists():
            fr = json.loads(frozen.read_text(encoding="utf-8"))
            w(f"- frozen selection: retained {fr['retained_devices']}, "
              f"excluded {fr['excluded_devices']} "
              f"(frozen_at {fr['frozen_at']})")
    else:
        w("- pilot config: NOT FOUND")

    runs = [collect_run(d) for d in run_dirs
            if (d / "run_config.json").exists()]
    if not runs:
        w("\nNo run directories with run_config.json were given -- "
          "this report is a shell until generation completes.\n")

    # ---- coverage ----------------------------------------------------
    w("\n## Completion and evaluation coverage\n")
    w("| run | model | expected pairs | attempted | eval records |")
    w("|---|---|---|---|---|")
    for r in runs:
        s = r["scored"]
        w(f"| {r['dir'].name} | {r['model']} | {s['expected_pairs']} "
          f"| {s['attempted_pairs']} | {len(r['evals'])} |")

    # ---- accuracy ----------------------------------------------------
    w("\n## Answer accuracy (all responses, compliant or not)\n")
    w("Primary accuracy uses every generated response; nothing is "
      "filtered to compliant responses.\n")
    w("| run | dataset | condition | correct/attempted | acc | parse "
      "failures | hit limit |")
    w("|---|---|---|---|---|---|---|")
    for r in runs:
        for cell, s in sorted(r["scored"]["summary"].items()):
            ds, cond = cell.split("/", 1)
            bad = {k: v for k, v in s["parse_status"].items()
                   if k != "parsed" and v}
            w(f"| {r['dir'].name} | {ds} | {cond} "
              f"| {s['correct']}/{s['attempted']} "
              f"| {s['accuracy_strict']} | {bad or '-'} "
              f"| {s['hit_output_limit']} |")

    # ---- matched deltas ----------------------------------------------
    w("\n## Matched accuracy differences vs controls (percentage "
      "points)\n")
    w("Each delta uses question IDs present in BOTH conditions; "
      "incomplete comparisons are flagged.\n")
    w("| run | dataset | condition | vs answer_only | vs "
      "prose_reasoning | vs line_breaks |")
    w("|---|---|---|---|---|---|")
    for r in runs:
        per = r["scored"]["per_response"]
        conds = {c for (_, c) in
                 [(k.split('/')[0], k.split('/')[1])
                  for k in r["scored"]["summary"]]}
        for ds in sorted({k.split('/')[0]
                          for k in r["scored"]["summary"]}):
            for cond in sorted(conds):
                if cond in CONTROL_ORDER:
                    continue
                deltas = []
                for ctrl in CONTROL_ORDER:
                    d = matched_delta(per, ds, cond, ctrl)
                    if d["delta_pp"] is None:
                        deltas.append("n/a")
                    else:
                        t = f"{d['delta_pp']:+.1f}"
                        if d["incomplete"]:
                            t += "*"
                        deltas.append(t)
                w(f"| {r['dir'].name} | {ds} | {cond} "
                  f"| {deltas[0]} | {deltas[1]} | {deltas[2]} |")
    w("\n\\* = incomplete match (fewer shared IDs than responses). "
      "A difference between adherence and accuracy does not show the "
      "constraint CAUSED it.\n")

    # ---- adherence ---------------------------------------------------
    w("\n## Evaluator-assessed adherence (poetic conditions)\n")
    w("| run | dataset | device | evaluated | pass | fail | uncertain "
      "| not_evaluable | processing issues | joint success "
      "(correct & pass) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        scored_pairs = {(p["example_id"], p["condition"]): p["correct"]
                        for p in r["scored"]["per_response"]}
        groups = defaultdict(list)
        for e in r["evals"]:
            groups[(e["dataset"], e["condition"])].append(e)
        for (ds, cond), es in sorted(groups.items()):
            auto = {}
            for e in es:
                if e["source"] in AUTO_SOURCE_ORDER:
                    auto[e["response_id"]] = auto_verdict(
                        r["by_rid"], e["response_id"])
            verdicts = [v["verdict"] for v in auto.values() if v]
            cnt = {v: verdicts.count(v) for v in ad.VERDICTS}
            proc = sum(1 for v in auto.values()
                       if v.get("processing_status") not in
                       (None, "resolved", "no_applicable_lines",
                        "missing_reasoning_view"))
            joint = sum(1 for rid_, v in auto.items()
                        if v and v["verdict"] == "pass"
                        and scored_pairs.get(
                            (rid_.split("::")[-1], cond)))
            w(f"| {r['dir'].name} | {ds} | {cond} | {len(verdicts)} "
              f"| {cnt['pass']} | {cnt['fail']} | {cnt['uncertain']} "
              f"| {cnt['not_evaluable']} | {proc} | {joint} |")
    w("\nSources: records carry a `source` field (heuristic / "
      "llm_judge / human_audit / adjudicated); automated estimates "
      "prefer llm_judge over heuristic per response.\n")

    # ---- validation & audit ------------------------------------------
    w("\n## Evaluator validation\n")
    val = REPO_ROOT / "reports" / "eval_validation.json"
    if val.exists():
        v = json.loads(val.read_text(encoding="utf-8"))
        o = v["overall"]
        w(f"- vs screening labels ({v['csv']}): resolved pairs "
          f"{o['resolved_pairs']}, agreement "
          f"{o['agreement_on_resolved']}, false_pass {o['false_pass']}, "
          f"false_fail {o['false_fail']} "
          "(screening labels were evaluator-development data -- "
          "not independent validation).")
    else:
        w("- screening-label validation: not run "
          "(scripts/validate_evaluator.py)")
    if audit_dir and (audit_dir / "evaluator_acceptance.json").exists():
        acc = json.loads((audit_dir / "evaluator_acceptance.json")
                         .read_text(encoding="utf-8"))
        w(f"- evaluator acceptance decisions: {acc['devices']}")
    else:
        w("- fresh blinded audit: not run (scripts/audit_pack.py)")

    # ---- costs -------------------------------------------------------
    w("\n## Costs (this pipeline's ledger only)\n")
    ledger = budget_mod.BudgetLedger(DEFAULT_BUDGET_CONFIG)
    st = ledger.state()
    w(f"- committed total: ${st['committed_usd']:.4f} "
      f"(cap ${ledger.cfg['total_cap_usd']:.2f}); "
      f"open reservations ${st['open_reserved_usd']:.4f}")
    for b, v in sorted(st["buckets"].items()):
        w(f"- bucket `{b}`: committed ${v['committed_usd']:.4f} "
          f"+ open ${v['open_reserved_usd']:.4f}")
    w("\nGeneration vs judging split: bucket 'judging' covers all LLM "
      "judge calls (validation + pilot, tagged by run_id in the "
      "ledger); every other bucket is generation.")

    # ---- limitations -------------------------------------------------
    w("\n## Limitations and unresolved issues\n")
    w("- Automated adherence is evaluator-assessed; until the blinded "
      "audit is done and per-device acceptance is recorded, treat every "
      "rate here as PROVISIONAL.")
    w("- Meter remains the least mechanical device: uncertain cases go "
      "to the judge, and unresolved cases are never silently counted "
      "as failures.")
    w("- If the judge shares a family with a generator, validation "
      "errors should be examined per source model (see "
      "configs/judge_*.json notes).")
    w("- Accuracy deltas are descriptive; no causal claim about "
      "adherence is made.")
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--pilot-config", type=Path,
                    default=REPO_ROOT / "configs" / "dev_pilot.json")
    ap.add_argument("--audit-dir", type=Path,
                    default=REPO_ROOT / "reviews" / "pilot_audit")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    text = report(args.run_dirs, args.pilot_config, args.audit_dir,
                  args.out)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
