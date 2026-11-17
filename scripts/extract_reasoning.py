#!/usr/bin/env python3
"""Backfill reasoning-extraction fields for an existing run.

Reads <run-dir>/responses.jsonl and writes a NEW derived file,
<run-dir>/responses.with_reasoning.jsonl, with the same records plus:
run_id, model, reasoning_text, reasoning_parse_status,
final_answer_text, final_answer_values, final_answer_status.

The original responses.jsonl is never modified and nothing is
regenerated -- extraction is ordinary string code over the saved raw
response (see response_parsing.py). Works on both current records
(raw_response) and older ones (response).

Usage:
    python3 scripts/extract_reasoning.py --run-dir outputs/pilot_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import prompt_lib
import response_parsing as rp

DERIVED_FILE = "responses.with_reasoning.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"defaults to <run-dir>/{DERIVED_FILE}")
    args = ap.parse_args()

    src = args.run_dir / "responses.jsonl"
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    config_path = args.run_dir / "run_config.json"
    run_id = args.run_dir.name
    model_meta = None
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        compat = cfg.get("compatibility", {})
        quant = (cfg.get("environment", {})
                    .get("model_config", {})
                    .get("quantization"))
        model_meta = {
            "repo": compat.get("model_repo"),
            "revision": compat.get("model_revision"),
            "quantization": quant,
        }
        run_id = cfg.get("run_name", run_id)

    out_path = args.out or args.run_dir / DERIVED_FILE
    templates = prompt_lib.load_templates()
    n = 0
    status_counts: dict[str, int] = {}
    with open(src, encoding="utf-8") as f, \
            open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            parsed = rp.extract_reasoning_fields(
                rp.record_raw_text(rec),
                prompt_lib.expects_reasoning(rec["condition"], templates),
            )
            rec.setdefault("run_id", run_id)
            rec.setdefault("model", model_meta)
            # Normalize the raw-response key on the derived copy only.
            if "raw_response" not in rec and "response" in rec:
                rec["raw_response"] = rec.pop("response")
            rec.update(parsed)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            s = parsed["reasoning_parse_status"]
            status_counts[s] = status_counts.get(s, 0) + 1

    print(f"wrote {n} derived records -> {out_path}")
    print(f"reasoning_parse_status counts: {status_counts}")
    print(f"original {src} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
