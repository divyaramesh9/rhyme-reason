#!/usr/bin/env python3
"""Offline cost preview for a planned generation run.

Renders the exact messages the run would send (real manifest examples +
real prompt templates) and prices them with the model config's rates.

Two estimates are printed:

  * assumed-output  -- output tokens set by --assumed-output-tokens
                       (your guess at a typical response length);
  * conservative    -- every response billed at the full
                       --max-new-tokens output cap (1024 default).

Input tokens are APPROXIMATE: the o200k tokenizer is not guaranteed
available offline, so estimate_input_tokens uses a deliberately
conservative character heuristic (chars/3 + ~4 tokens/message + ~3
priming). No caching or batch discounts are assumed. This previews what
the budget ledger would allow, not the account's real balance.

Usage:
    python3 scripts/estimate_cost.py \
        --manifest data/manifests/screening4.json \
        --conditions rhyme_scheme syllable_counts alliteration anaphora \
            epistrophe meter couplets stanza_structures \
        --model-config configs/model_gpt-4o-2024-08-06.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import backends
import prompt_lib
import run_generation as rg

REPO_ROOT = Path(__file__).resolve().parent.parent


def preview(manifest: Path, conditions: list[str], model_cfg: dict,
            prompts_dir: Path, processed_dir: Path,
            limit_per_dataset: int | None, max_new_tokens: int,
            assumed_output_tokens: int) -> dict:
    templates = prompt_lib.load_templates(prompts_dir)
    examples = rg.load_manifest_examples(manifest, limit_per_dataset,
                                         processed_dir)
    items = []
    total_in = 0
    for rec in examples:
        for cond in conditions:
            msgs = prompt_lib.build_messages(rec, cond, templates)
            est = backends.estimate_input_tokens(msgs)
            total_in += est
            items.append({"example_id": rec["example_id"],
                          "dataset": rec["dataset"],
                          "condition": cond,
                          "est_input_tokens": est})
    n = len(items)
    p = model_cfg.get("pricing") or {}
    in_rate = p.get("input_usd_per_1m", 0.0)
    out_rate = p.get("output_usd_per_1m", 0.0)
    return {
        "model": model_cfg.get("model") or model_cfg.get("repo_id"),
        "backend": model_cfg.get("backend"),
        "n_responses": n,
        "est_input_tokens_total": total_in,
        "assumed_output_tokens_each": assumed_output_tokens,
        "max_output_tokens_each": max_new_tokens,
        "cost_assumed_usd": round(
            total_in * in_rate / 1e6
            + n * assumed_output_tokens * out_rate / 1e6, 4),
        "cost_conservative_usd": round(
            total_in * in_rate / 1e6
            + n * max_new_tokens * out_rate / 1e6, 4),
        "items": items,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--model-config", type=Path,
                    default=rg.DEFAULT_MODEL_CONFIG)
    ap.add_argument("--prompts-dir", type=Path,
                    default=prompt_lib.PROMPT_DIR)
    ap.add_argument("--processed-dir", type=Path,
                    default=rg.PROCESSED_DIR)
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int,
                    default=1024)
    ap.add_argument("--assumed-output-tokens", type=int, default=150,
                    help="labeled guess at typical response length for "
                         "the non-conservative estimate")
    args = ap.parse_args()

    available = prompt_lib.available_conditions(args.prompts_dir)
    bad = [c for c in args.conditions if c not in available]
    if bad:
        raise SystemExit(f"error: unknown condition(s) {bad}; "
                         f"available: {available}")

    model_cfg = backends.load_model_config(args.model_config)
    rep = preview(args.manifest, args.conditions, model_cfg,
                  args.prompts_dir, args.processed_dir,
                  args.limit_per_dataset, args.max_new_tokens,
                  args.assumed_output_tokens)

    print(f"model: {rep['model']}  (backend: {rep['backend']})")
    print(f"planned responses: {rep['n_responses']} "
          f"({len(args.conditions)} condition(s))")
    if rep["backend"] != "openai":
        print("no paid provider -- estimated cost $0.00 "
              "(local inference).")
        return 0

    p = model_cfg["pricing"]
    print(f"pricing: ${p['input_usd_per_1m']}/1M input, "
          f"${p['output_usd_per_1m']}/1M output "
          f"({p.get('source')}, verified {p.get('verified_utc')})")
    print(f"estimated input tokens total: ~{rep['est_input_tokens_total']} "
          "(APPROXIMATE -- conservative chars/3 heuristic + per-message "
          "overhead, no tokenizer available offline)")
    print(f"cost if every response = {rep['assumed_output_tokens_each']} "
          f"output tokens (assumed length): "
          f"${rep['cost_assumed_usd']:.4f}")
    print(f"cost if every response hits the {rep['max_output_tokens_each']}"
          f"-token cap (conservative): "
          f"${rep['cost_conservative_usd']:.4f}")
    print("no caching/batch discounts assumed; ledger reservations use "
          "the conservative figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
