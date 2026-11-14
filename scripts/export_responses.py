#!/usr/bin/env python3
"""Export each saved response to a readable UTF-8 .txt file.

Reads the canonical <run-dir>/responses.jsonl (and
responses.with_reasoning.jsonl if present) and writes one file per
response under:

    <export-root>/<run_id>/<dataset>/<condition>/
        <model-tag>__<dataset>__<condition>__<example_id>.txt

The filename embeds model, dataset, condition, and example_id so files
can never collide across runs of different models or conditions.

Each file carries the example ID, dataset, condition, run, model
reference, finish info, the question as the model saw it, and the
complete raw response -- enough context to inspect the visible
reasoning by hand.

Usage:
    python3 scripts/export_responses.py --run-dir outputs/pilot_baseline
    python3 scripts/export_responses.py --run-dir outputs/pilot_baseline \
        --export-root exports
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import prompt_lib
import response_parsing as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def safe_tag(text: str) -> str:
    """Filesystem-safe tag: letters/digits/dots/dashes/underscores."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def model_tag(model: dict | None, compat: dict) -> str:
    repo = (model or {}).get("repo") or compat.get("model_repo") or "model"
    rev = (model or {}).get("revision") or compat.get("model_revision") or ""
    return safe_tag(f"{repo.split('/')[-1]}-{rev[:8]}" if rev else repo)


def load_questions(run_dir: Path, processed_dir: Path) -> dict:
    """example_id -> model-visible question block, via the run's manifest."""
    cfg = json.loads((run_dir / "run_config.json").read_text(
        encoding="utf-8"))
    manifest_path = Path(cfg["compatibility"]["manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = {}
    for dataset, pool in manifest["pools"].items():
        inputs = processed_dir / f"{dataset}_{pool['split']}.inputs.jsonl"
        by_id = {}
        with open(inputs, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    by_id[r["example_id"]] = r
        for eid in pool["example_ids"]:
            if eid in by_id:
                questions[eid] = prompt_lib.question_block(by_id[eid])
    return questions


def render_txt(rec: dict, question: str | None,
               model: dict | None, compat: dict, run_id: str) -> str:
    raw = rp.record_raw_text(rec)
    model_line = "unavailable"
    if model and model.get("repo"):
        model_line = f"{model['repo']}@{model.get('revision')}"
        if model.get("quantization"):
            model_line += f" (quantization {model['quantization']})"
    rps = rec.get("reasoning_parse_status")
    if rps is None:
        # Older records predate this field; compute it for display.
        rps = rp.extract_reasoning_fields(
            raw, rec.get("condition") != "answer_only"
        )["reasoning_parse_status"] + " (derived)"
    parts = [
        f"example_id: {rec.get('example_id')}",
        f"dataset: {rec.get('dataset')}",
        f"condition: {rec.get('condition')}",
        f"run_id: {rec.get('run_id') or run_id}",
        f"model: {model_line}",
        f"finish_reason: {rec.get('finish_reason')}",
        f"generated_tokens: {rec.get('generated_tokens')}",
        f"hit_output_limit: {rec.get('hit_output_limit')}",
        f"reasoning_parse_status: {rps}",
        "",
        "=== QUESTION (as shown to the model) ===",
        question if question is not None else "(question unavailable)",
        "",
        "=== RAW RESPONSE ===",
        raw,
        "",
    ]
    return "\n".join(str(p) for p in parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--export-root", type=Path,
                    default=REPO_ROOT / "exports")
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()

    src = args.run_dir / "responses.jsonl"
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    cfg = json.loads((args.run_dir / "run_config.json").read_text(
        encoding="utf-8"))
    compat = cfg.get("compatibility", {})
    run_id = cfg.get("run_name", args.run_dir.name)
    env_model = cfg.get("environment", {}).get("model_config", {})
    model = {
        "repo": compat.get("model_repo"),
        "revision": compat.get("model_revision"),
        "quantization": env_model.get("quantization"),
    }
    tag = model_tag(model, compat)
    questions = load_questions(args.run_dir, args.processed_dir)

    n = 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec_model = rec.get("model") or model
            fname = (
                f"{tag}__{safe_tag(rec['dataset'])}__"
                f"{safe_tag(rec['condition'])}__"
                f"{safe_tag(rec['example_id'])}.txt"
            )
            dest = (args.export_root / run_id / rec["dataset"]
                    / rec["condition"] / fname)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                render_txt(rec, questions.get(rec["example_id"]),
                           rec_model, compat, run_id),
                encoding="utf-8",
            )
            n += 1
    print(f"exported {n} response file(s) -> "
          f"{args.export_root / run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
