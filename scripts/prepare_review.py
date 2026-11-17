#!/usr/bin/env python3
"""Create a blank manual-adherence review pack for one or more runs.

For every saved response this writes:

  * a readable review file containing the question, the exact device
    instructions (the condition template, verbatim), and the FULL raw
    response with its formatting untouched, followed by a clearly
    separated diagnostic line-numbered view;
  * one row in a review CSV keyed by a stable response_id, with the
    device rule text, a path to that review file, and EMPTY
    adherence_label / notes columns for the human reviewer.

No adherence is inferred or prefilled. Gold answers never appear in
review materials; the model's own final answer remains inside its raw
response. Allowed labels: pass | fail | uncertain | not_evaluable.

Usage:
    python3 scripts/prepare_review.py \
        --run-dirs outputs/screening_qwen --out-dir reviews/screening_qwen
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import common
import export_responses as ex
import inspect_response as insp
import response_parsing as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROMPTS_DIR = REPO_ROOT / "prompts"

CSV_FIELDS = [
    "response_id", "run_id", "model", "dataset", "example_id",
    "condition", "device_rule_file", "device_rule_sha256",
    "device_rule_text", "response_file", "adherence_label", "notes",
]


def response_id(run_id: str, model: dict, compat: dict,
                rec: dict) -> str:
    return "::".join([
        run_id,
        common.model_tag(model, compat),
        rec["dataset"],
        rec["condition"],
        rec["example_id"],
    ])


def rule_text_for(condition: str, prompts_dir: Path) -> tuple[str, str]:
    """(relative file path, sha256) of the exact device-rule template."""
    p = prompts_dir / f"{condition}.txt"
    text = p.read_text(encoding="utf-8")
    try:
        rel = str(p.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(p)
    return rel, hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_review(rec: dict, question: str | None, model: dict,
                  compat: dict, run_id: str,
                  prompts_dir: Path) -> str:
    raw = rp.record_raw_text(rec)
    rule_file, _ = rule_text_for(rec["condition"], prompts_dir)
    rule = (prompts_dir / f"{rec['condition']}.txt").read_text(
        encoding="utf-8")
    model_line = "unavailable"
    if model and model.get("repo"):
        model_line = f"{model['repo']}@{model.get('revision')}"
        if model.get("quantization"):
            model_line += f" (quantization {model['quantization']})"
    parts = [
        "MANUAL ADHERENCE REVIEW -- fill in the CSV row for this "
        "response_id.",
        "Judge ONLY whether the visible reasoning follows the device "
        "rule. Do not judge answer correctness.",
        "",
        f"response_id: {response_id(run_id, model, compat, rec)}",
        f"run_id: {run_id}",
        f"model: {model_line}",
        f"dataset: {rec['dataset']}",
        f"example_id: {rec['example_id']}",
        f"condition: {rec['condition']}",
        f"finish_reason: {rec.get('finish_reason')}  "
        f"hit_output_limit: {rec.get('hit_output_limit')}",
        "",
        f"=== DEVICE INSTRUCTIONS ({rule_file}, verbatim) ===",
        rule,
        "",
        "=== QUESTION (as shown to the model) ===",
        question if question is not None else "(question unavailable)",
        "",
        "=== RAW RESPONSE (unaltered) ===",
        raw,
        "",
        "=== DIAGNOSTIC: line-numbered view (NOT part of the response) "
        "===",
        insp.numbered_view(raw),
        "",
    ]
    return "\n".join(str(p) for p in parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="review pack directory (CSV + files)")
    ap.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "review.csv"
    rows = []
    n_files = 0
    for run_dir in args.run_dirs:
        src = run_dir / "responses.jsonl"
        if not src.exists():
            print(f"error: {src} not found", file=sys.stderr)
            return 1
        cfg = json.loads((run_dir / "run_config.json").read_text(
            encoding="utf-8"))
        compat = cfg.get("compatibility", {})
        run_id = cfg.get("run_name", run_dir.name)
        env_model = cfg.get("environment", {}).get("model_config", {})
        fallback_model = {
            "repo": compat.get("model_repo"),
            "revision": compat.get("model_revision"),
            "quantization": env_model.get("quantization"),
        }
        questions = ex.load_questions(run_dir, args.processed_dir)
        for rec in common.read_jsonl(src):
            model = rec.get("model") or fallback_model
            rid = response_id(run_id, model, compat, rec)
            fname = (
                f"{common.model_tag(model, compat)}__"
                f"{common.safe_tag(rec['dataset'])}__"
                f"{common.safe_tag(rec['condition'])}__"
                f"{common.safe_tag(rec['example_id'])}.review.txt")
            dest = (args.out_dir / "files" / run_id / rec["dataset"]
                    / rec["condition"] / fname)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(render_review(
                rec, questions.get(rec["example_id"]), model, compat,
                run_id, args.prompts_dir), encoding="utf-8")
            n_files += 1
            rule_file, rule_sha = rule_text_for(
                rec["condition"], args.prompts_dir)
            rule = (args.prompts_dir / f"{rec['condition']}.txt"
                    ).read_text(encoding="utf-8")
            rows.append({
                "response_id": rid,
                "run_id": run_id,
                "model": f"{model.get('repo')}@{model.get('revision')}",
                "dataset": rec["dataset"],
                "example_id": rec["example_id"],
                "condition": rec["condition"],
                "device_rule_file": rule_file,
                "device_rule_sha256": rule_sha,
                "device_rule_text": rule,
                "response_file": str(dest),
                "adherence_label": "",
                "notes": "",
            })

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} blank review row(s) -> {csv_path}")
    print(f"wrote {n_files} readable review file(s) -> "
          f"{args.out_dir / 'files'}")
    print("labels are EMPTY -- fill pass|fail|uncertain|not_evaluable, "
          "then: python3 scripts/import_review.py --csv " + str(csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
