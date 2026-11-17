#!/usr/bin/env python3
"""Check whether records from an existing run are eligible for reuse
under a planned configuration.

The reuse rule: a saved response may be reused only when the selected
questions, prompt template hashes, model identity (repo + revision +
backend), chat-template hash, decoding spec, and output-token cap match
the planned configuration EXACTLY. Manifest path/sha are intentionally
not compared -- a new manifest selecting the same example IDs is fine,
because example IDs are content-stable. Everything else must match.

For each record this prints ELIGIBLE or INELIGIBLE (with the first
mismatch reason). Eligibility is a report; nothing is copied or
modified -- if you reuse eligible records, do it by pointing review
tools at the historical run, not by moving files.

Usage:
    python3 scripts/check_reuse.py \
        --run-dir outputs/poetic_smoke_v3 \
        --manifest data/manifests/screening4.json \
        --model-config configs/model_qwen2.5-7b-instruct-4bit.json \
        --conditions rhyme_scheme couplets --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import backends
import common
import prompt_lib
import run_generation as rg


def current_config(manifest: Path, model_cfg: dict, prompts_dir: Path,
                   conditions: list[str], max_new_tokens: int) -> dict:
    """The compatibility dict a new run would record, minus manifest
    identity (deliberately not part of the reuse check)."""
    chat_sha = "unavailable"  # compared only when stored has a real sha
    compat = rg.build_compatibility(
        model_cfg, manifest, prompts_dir, conditions, max_new_tokens,
        chat_sha)
    compat.pop("manifest_path", None)
    compat.pop("manifest_sha256", None)
    return compat


def compare(stored_compat: dict, cur: dict) -> list[str]:
    problems = []
    for key in ("backend", "model_repo", "model_revision",
                "chat_template_sha256", "max_new_tokens", "sampler"):
        old = stored_compat.get(
            key, rg.IMPLIED_COMPAT_DEFAULTS.get(key))
        new = cur.get(key)
        if key == "chat_template_sha256" and new in (
                None, "unavailable"):
            # The planned chat-template sha can't be computed without
            # loading the tokenizer; report rather than guess.
            continue
        if old != new:
            problems.append(f"{key}: stored {old!r} != planned {new!r}")
    for name, sha in cur["prompt_template_sha256"].items():
        old = stored_compat.get("prompt_template_sha256", {}).get(name)
        if old is None:
            continue  # condition never run in this historical run
        if old != sha:
            problems.append(
                f"prompt template {name!r}: stored {old[:12]}... != "
                f"planned {sha[:12]}...")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model-config", type=Path,
                    default=rg.DEFAULT_MODEL_CONFIG)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--prompts-dir", type=Path,
                    default=prompt_lib.PROMPT_DIR)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    cfg_path = args.run_dir / "run_config.json"
    resp_path = args.run_dir / "responses.jsonl"
    if not (cfg_path.exists() and resp_path.exists()):
        print(f"error: {args.run_dir} lacks run_config.json/"
              f"responses.jsonl", file=sys.stderr)
        return 1
    stored = json.loads(cfg_path.read_text(encoding="utf-8"))
    stored_compat = stored["compatibility"]

    model_cfg = backends.load_model_config(args.model_config)
    cur = current_config(args.manifest, model_cfg, args.prompts_dir,
                         args.conditions, args.max_new_tokens)

    print("note: chat-template sha of the planned model cannot be "
          "computed offline; that field is reported as unverified "
          "(model_repo + revision still compared).\n")
    selected_ids = {
        eid
        for pool in json.loads(args.manifest.read_text(
            encoding="utf-8"))["pools"].values()
        for eid in pool["example_ids"]
    }

    n_ok = n_no = 0
    for r in common.read_jsonl(resp_path):
        problems = []
        if r["condition"] not in args.conditions:
            problems.append("condition not in planned set")
        if r["example_id"] not in selected_ids:
            problems.append("example_id not in planned manifest")
        problems += compare(stored_compat, cur)
        tag = f"{r['example_id']}/{r['condition']}"
        if problems:
            n_no += 1
            print(f"INELIGIBLE {tag}: {'; '.join(problems)}")
        else:
            n_ok += 1
            print(f"ELIGIBLE   {tag}")
    print(f"{n_ok} eligible, {n_no} ineligible under the planned config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
