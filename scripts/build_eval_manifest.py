#!/usr/bin/env python3
"""Build a held-out evaluation manifest from the official test splits.

The screening/dev questions come from gsm8k train + logiqa_v2
validation; evaluation questions come from the official test splits, so
they are disjoint by construction at the split level. Because LogiQA
contains exact-duplicate items ACROSS official splits (see
reports/data_validation.json), disjointness is additionally enforced by
CONTENT, not just IDs:

  * gsm8k:     the exact question string;
  * logiqa_v2: (passage, question, ordered A-D option texts).

Any test item whose content key occurs ANYWHERE in that dataset's dev
pool split (gsm8k train / logiqa_v2 validation) is excluded, which
covers the screening4, pilot, and pilot_disjoint selections and any
future dev use of those pools. Items are never deleted or modified;
the exclusion counts are recorded in the manifest.

Deterministic sample: random.Random(seed).sample over the sorted
eligible test IDs. The sample size is REQUIRED (--size-per-dataset) --
this script does not invent one. Preview the cost of a candidate size
with scripts/estimate_cost.py before generating.

Usage:
    python3 scripts/build_eval_manifest.py --size-per-dataset 50 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import select_disjoint_pilot as sdp

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

# dataset -> (evaluation split, dev pool split whose content must not
# appear in the evaluation set)
POOLS = {
    "gsm8k": {"eval_split": "test", "dev_split": "train"},
    "logiqa_v2": {"eval_split": "test", "dev_split": "validation"},
}
DATASET_ORDER = ("gsm8k", "logiqa_v2")


def build_eval_manifest(processed_dir: Path, n: int, seed: int,
                        name: str) -> dict:
    pools = {}
    problems = []
    for dataset in DATASET_ORDER:
        spec = POOLS[dataset]
        test_records = sdp.load_jsonl(
            processed_dir / f"{dataset}_{spec['eval_split']}.inputs.jsonl")
        dev_keys = {
            sdp.content_key(r)
            for r in sdp.load_jsonl(
                processed_dir / f"{dataset}_{spec['dev_split']}.inputs.jsonl")
        }
        eligible, excluded = [], []
        for r in test_records:
            if sdp.content_key(r) in dev_keys:
                excluded.append(r["example_id"])
            else:
                eligible.append(r["example_id"])
        chosen = sorted(random.Random(seed).sample(
            sorted(eligible), min(n, len(eligible))))
        if len(chosen) < n:
            problems.append(
                f"{dataset}: requested {n} but only {len(chosen)} "
                f"eligible test items remain after content exclusion")
        pools[dataset] = {
            "split": spec["eval_split"],
            "eval_pool_size": len(test_records),
            "excluded_for_dev_pool_content_overlap": len(excluded),
            "eligible_count": len(eligible),
            "example_ids": chosen,
            "selected_count": len(chosen),
        }
    return {
        "name": name,
        "stage": "evaluation",
        "purpose": ("Held-out final-evaluation set drawn from the "
                    "official test splits. Content-disjoint from all "
                    "development pools (screening, pilots) by exact "
                    "question content, not just IDs."),
        "seed": seed,
        "sample_size_per_dataset": n,
        "matching_rule": sdp.MATCHING_RULE,
        "selection_procedure": (
            "For each dataset, drop every test-split item whose content "
            "key occurs anywhere in the corresponding dev pool split "
            "(gsm8k train / logiqa_v2 validation). From the remaining "
            "eligible test IDs, take random.Random(seed).sample over the "
            "sorted IDs."),
        "pools": pools,
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size-per-dataset", type=int, required=True,
                    help="number of test items per dataset (required; "
                         "preview cost with estimate_cost.py first)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None,
                    help="manifest name (default eval_n<N>_seed<S>)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()

    if args.size_per_dataset <= 0:
        ap.error("--size-per-dataset must be positive")

    name = args.name or f"eval_n{args.size_per_dataset}_seed{args.seed}"
    out = args.out or MANIFEST_DIR / f"{name}.json"

    manifest = build_eval_manifest(args.processed_dir,
                                   args.size_per_dataset, args.seed, name)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                   + "\n", encoding="utf-8")
    for dataset, pool in manifest["pools"].items():
        print(f"{dataset}: {pool['selected_count']} selected of "
              f"{pool['eligible_count']} eligible "
              f"({pool['excluded_for_dev_pool_content_overlap']} test items "
              f"excluded for dev-pool content overlap; pool "
              f"{pool['eval_pool_size']})")
    print(f"wrote {out}")
    if manifest["problems"]:
        for p in manifest["problems"]:
            print(f"WARNING: {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
