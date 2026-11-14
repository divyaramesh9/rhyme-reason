#!/usr/bin/env python3
"""Score generated responses against gold answers.

Joins a run's responses.jsonl / failures.jsonl with the processed
*.answers.jsonl files on example_id (this is the only script that reads
answer files; generation never sees them).

Only the designated FINAL_ANSWER: line is parsed -- numbers or letters
appearing anywhere else in the response are ignored. Missing, malformed,
or ambiguous finals count as incorrect (they are never dropped from the
denominator), and generation failures count as attempted-and-incorrect.

Usage:
    python3 scripts/score_answers.py --run-dir outputs/pilot_baseline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import response_parsing as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

FINAL_ANSWER_RE = rp.FINAL_ANSWER_LINE_RE
extract_final_answer = rp.extract_final_answer

NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
LETTER_RE = re.compile(r"^[ABCD]\.?$", re.IGNORECASE)

PARSE_STATUSES = ("parsed", "missing", "malformed", "ambiguous")


# --------------------------------------------------------------------------
# Final-answer extraction and normalization
# --------------------------------------------------------------------------


def normalize_gsm8k(text: str) -> Fraction | None:
    """Normalize harmless formatting (whitespace, $ , thousands commas,
    one trailing period) to an exact numeric value, else None."""
    s = text.strip()
    s = s.lstrip("$").strip()
    s = s.replace(",", "")
    s = re.sub(r"\.$", "", s)  # "72." -> "72"
    if not NUMERIC_RE.fullmatch(s):
        return None
    return Fraction(s)


def normalize_logiqa(text: str) -> str | None:
    """Accept an unambiguous A/B/C/D label (any case, harmless whitespace,
    optional single trailing period); else None."""
    s = text.strip().upper()
    if LETTER_RE.fullmatch(s):
        return s.rstrip(".")
    return None


def score_response(dataset: str, raw: str, gold: str,
                   condition: str) -> dict:
    """Score one raw response. Never repairs or regenerates anything."""
    status, values = extract_final_answer(raw)
    result = {
        "parse_status": "missing" if status == "missing" else status,
        "parsed_values": values,
        "normalized_prediction": None,
        "gold": gold,
        "correct": False,
        "extra_text": None,
    }
    if status != "ok":
        return result

    if condition == "answer_only":
        remainder = FINAL_ANSWER_RE.sub("", raw).strip()
        result["extra_text"] = bool(remainder)

    if dataset == "gsm8k":
        pred = normalize_gsm8k(values[0])
        gold_v = normalize_gsm8k(gold)
        if pred is None:
            result["parse_status"] = "malformed"
        else:
            result["parse_status"] = "parsed"
            result["normalized_prediction"] = str(pred)
            result["correct"] = pred == gold_v
    elif dataset == "logiqa_v2":
        pred = normalize_logiqa(values[0])
        if pred is None:
            result["parse_status"] = "malformed"
        else:
            result["parse_status"] = "parsed"
            result["normalized_prediction"] = pred
            result["correct"] = pred == gold
    else:
        raise KeyError(f"no scoring rule for dataset {dataset!r}")
    return result


# --------------------------------------------------------------------------
# Run-level scoring
# --------------------------------------------------------------------------

def load_answers(manifest_path: Path) -> dict[str, dict]:
    """example_id -> {gold_answer, dataset} for manifest examples only."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {}
    for dataset, pool in manifest["pools"].items():
        path = PROCESSED_DIR / f"{dataset}_{pool['split']}.answers.jsonl"
        with open(path, encoding="utf-8") as f:
            by_id = {
                json.loads(l)["example_id"]: json.loads(l)
                for l in f if l.strip()
            }
        for eid in pool["example_ids"]:
            if eid in by_id:
                out[eid] = {
                    "gold_answer": by_id[eid]["gold_answer"],
                    "dataset": dataset,
                }
    return out


def score_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text(
        encoding="utf-8"))
    compat = config["compatibility"]
    conditions = compat["conditions"]
    manifest_path = REPO_ROOT / compat["manifest_path"]
    answers = load_answers(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (eid, c)
        for pool in manifest["pools"].values()
        for eid in pool["example_ids"]
        for c in conditions
    }

    def _load(name):
        # Iterate file lines (LF only): response text may contain Unicode
        # line separators that splitlines() would wrongly split on.
        path = run_dir / name
        records = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        return records

    responses = _load("responses.jsonl")
    failures = _load("failures.jsonl")

    # One scored record per (example_id, condition) response; later
    # duplicate rows (shouldn't happen) are scored individually and counted.
    per_response = []
    for r in responses:
        eid, cond = r["example_id"], r["condition"]
        gold = answers.get(eid)
        scored = score_response(
            r["dataset"], rp.record_raw_text(r), gold["gold_answer"], cond
        ) if gold else None
        per_response.append({
            "example_id": eid,
            "dataset": r["dataset"],
            "condition": cond,
            "hit_output_limit": r.get("hit_output_limit"),
            "generated_tokens": r.get("generated_tokens"),
            "elapsed_seconds": r.get("elapsed_seconds"),
            "reasoning_parse_status": r.get("reasoning_parse_status"),
            "scored": scored is not None,
            **(scored or {"parse_status": "no_gold", "correct": False,
                          "normalized_prediction": None,
                          "gold": None, "extra_text": None}),
        })

    attempted = {(r["example_id"], r["condition"]) for r in responses}
    attempted |= {(r["example_id"], r["condition"]) for r in failures}
    not_attempted = sorted(expected - attempted)

    # Denominator = unique (example_id, condition) pairs: a pair that
    # failed once and later succeeded counts once, as a response.
    responded_pairs = {(r["example_id"], r["condition"])
                       for r in responses}
    deduped = {}
    for r in per_response:
        deduped[(r["example_id"], r["condition"])] = r
    per_pair = list(deduped.values())
    failed_pairs = {(f["example_id"], f["condition"]) for f in failures}
    failed_only = sorted(failed_pairs - responded_pairs)
    failed_meta = {}
    for f in failures:
        failed_meta.setdefault((f["example_id"], f["condition"]), f)

    cells = {(r["dataset"], r["condition"]) for r in per_pair}
    cells |= {(failed_meta[p]["dataset"], p[1]) for p in failed_only}
    summary = {}
    for dataset, cond in sorted(cells):
        rs = [r for r in per_pair
              if r["dataset"] == dataset and r["condition"] == cond]
        fs = [p for p in failed_only
              if failed_meta[p]["dataset"] == dataset and p[1] == cond]
        n = len(rs) + len(fs)
        correct = sum(1 for r in rs if r["correct"])
        summary[f"{dataset}/{cond}"] = {
            "attempted": n,
            "responses": len(rs),
            "generation_failures": len(fs),
            "correct": correct,
            "accuracy_strict": round(correct / n, 4) if n else None,
            "parse_status": {
                s: sum(1 for r in rs if r["parse_status"] == s)
                for s in PARSE_STATUSES + ("no_gold",)
            },
            "hit_output_limit": sum(
                1 for r in rs if r["hit_output_limit"]),
            "answer_only_extra_text": sum(
                1 for r in rs if r["extra_text"]) if cond == "answer_only"
                else None,
        }

    return {
        "run_dir": str(run_dir),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "conditions": conditions,
        "manifest": compat["manifest_path"],
        "expected_pairs": len(expected),
        "attempted_pairs": len(attempted),
        "not_attempted": not_attempted,
        "summary": summary,
        "per_response": per_response,
        "generation_failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <run-dir>/scores.json")
    args = ap.parse_args()

    report = score_run(args.run_dir)
    out = args.out or args.run_dir / "scores.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    print(f"run: {args.run_dir}  "
          f"({report['attempted_pairs']}/{report['expected_pairs']} "
          "pairs attempted)")
    for cell, s in report["summary"].items():
        print(f"  {cell}: {s['correct']}/{s['attempted']} correct "
              f"(strict acc {s['accuracy_strict']}); "
              f"parse={s['parse_status']}; "
              f"gen_failures={s['generation_failures']}; "
              f"hit_limit={s['hit_output_limit']}"
              + (f"; extra_text={s['answer_only_extra_text']}"
                 if s["answer_only_extra_text"] is not None else ""))
    if report["not_attempted"]:
        print(f"  not attempted yet: {len(report['not_attempted'])} pairs")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
