#!/usr/bin/env python3
"""Build the test-disjoint pilot manifest for the generation pilot.

Part 1 (scripts/prepare_data.py) reported exact-duplicate questions inside
LogiQA v2 and across its official splits (see reports/data_validation.json,
check "duplicate_questions"). The original data/manifests/pilot.json was
selected before that overlap was accounted for. This script derives a
separate manifest, data/manifests/pilot_disjoint.json, that additionally
guarantees, for every selected example:

  1. No exact duplicate question content inside the selected sample.
  2. Its question content does not occur in that dataset's official test
     split.

Matching rule (identical to prepare_data.check_duplicate_questions):
  * gsm8k:      the normalized "question" string.
  * logiqa_v2:  (context, question, option texts in A,B,C,D order).
The non-unique upstream LogiQA "id" / source_id is never used for matching.

Procedure per dataset:
  * Keep each original pilot ID (manifest order) whose key is absent from
    the test split and not already kept (first occurrence wins on
    within-sample duplicates).
  * Refill any freed slots with random.Random(seed).sample() over the
    sorted remaining eligible pool IDs -- the same selection style as the
    original pilot, so replacements are deterministic.

Original and processed splits are preserved unchanged; only the new
manifest is written. Official test inputs are read programmatically for
key-set membership only; test questions/answers are never printed or used.

Usage:
    python3 scripts/select_disjoint_pilot.py            # seed 42, size 20
    python3 scripts/select_disjoint_pilot.py --seed 7 --pilot-size 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import common

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
ORIGINAL_PILOT_PATH = MANIFEST_DIR / "pilot.json"
DISJOINT_PILOT_PATH = MANIFEST_DIR / "pilot_disjoint.json"

# Pilot pools: dataset -> (pool split, test split). Mirrors prepare_data.py.
POOLS = {
    "gsm8k": {"pool_split": "train", "test_split": "test"},
    "logiqa_v2": {"pool_split": "validation", "test_split": "test"},
}
DATASET_ORDER = ("gsm8k", "logiqa_v2")

MATCHING_RULE = (
    "gsm8k: exact question string; logiqa_v2: (context, question, option "
    "texts in A,B,C,D label order). The non-unique upstream source_id is "
    "never used."
)


def content_key(record: dict) -> tuple:
    """Exact-duplicate matching key; must stay in sync with
    prepare_data.check_duplicate_questions."""
    if record["dataset"] == "logiqa_v2":
        return (
            "logiqa_v2",
            record["context"],
            record["question"],
            tuple(record["options"][k] for k in sorted(record["options"])),
        )
    return ("gsm8k", record["question"])


def split_path(dataset: str, split: str) -> Path:
    return PROCESSED_DIR / f"{dataset}_{split}.inputs.jsonl"


def select_disjoint_ids(
    pool_records: dict[str, dict],
    test_keys: set,
    original_ids: list[str],
    n: int,
    seed: int,
) -> dict:
    """Pick n pool IDs disjoint from test content and internally unique.

    Returns the selection plus an audit trail of exclusions/replacements.
    """
    kept: list[str] = []
    kept_keys = set()
    excluded: list[dict] = []

    for eid in original_ids:
        if eid not in pool_records:
            excluded.append({"example_id": eid, "reason": "missing_from_pool"})
            continue
        k = content_key(pool_records[eid])
        if k in test_keys:
            excluded.append(
                {"example_id": eid, "reason": "content_occurs_in_test_split"}
            )
        elif k in kept_keys:
            excluded.append(
                {"example_id": eid, "reason": "in_sample_duplicate_content"}
            )
        else:
            kept.append(eid)
            kept_keys.add(k)

    replacements: list[str] = []
    needed = n - len(kept)
    eligible_total = 0
    if needed > 0:
        taken = set(kept) | {e["example_id"] for e in excluded}
        candidates = sorted(
            eid
            for eid, rec in pool_records.items()
            if eid not in taken
            and content_key(rec) not in test_keys
            and content_key(rec) not in kept_keys
        )
        eligible_total = len(candidates)
        replacements = sorted(
            random.Random(seed).sample(candidates, min(needed, eligible_total))
        )
        for eid in replacements:
            kept_keys.add(content_key(pool_records[eid]))

    return {
        "example_ids": sorted(set(kept) | set(replacements)),
        "kept_from_original": kept,
        "excluded": excluded,
        "replacements": replacements,
        "shortfall": max(0, n - len(kept) - len(replacements)),
        "eligible_remaining_after_keep": eligible_total if needed > 0 else None,
    }


def build_disjoint_pilot(
    processed_dir: Path, original_pilot: dict, n: int, seed: int
) -> dict:
    pools: dict[str, dict] = {}
    problems: list[str] = []

    for dataset in DATASET_ORDER:
        spec = POOLS[dataset]
        pool = {
            r["example_id"]: r
            for r in common.read_jsonl(split_path(dataset, spec["pool_split"]))
        }
        test_keys = {
            content_key(r)
            for r in common.read_jsonl(split_path(dataset, spec["test_split"]))
        }
        original_ids = original_pilot["pools"][dataset]["example_ids"]
        sel = select_disjoint_ids(pool, test_keys, original_ids, n, seed)

        pool_test_overlap = sum(
            1 for r in pool.values() if content_key(r) in test_keys
        )
        if sel["shortfall"]:
            problems.append(
                f"{dataset}: only {len(sel['example_ids'])} eligible example(s) "
                f"available, requested {n}"
            )

        pools[dataset] = {
            "split": spec["pool_split"],
            "pool_size": len(pool),
            "pool_items_whose_content_occurs_in_test": pool_test_overlap,
            "kept_from_original_pilot": sel["kept_from_original"],
            "excluded_from_original_pilot": sel["excluded"],
            "replacements": sel["replacements"],
            "example_ids": sel["example_ids"],
            "selected_count": len(sel["example_ids"]),
        }

    return {
        "name": "pilot_disjoint",
        "purpose": (
            "Development pilot for generation conditions. Same role as "
            "pilot.json, additionally guaranteeing no exact-duplicate "
            "content within the sample and no content overlap with the "
            "official test split. Derived from, and preserving, "
            "data/manifests/pilot.json."
        ),
        "seed": seed,
        "sample_size_per_dataset": n,
        "derived_from": "data/manifests/pilot.json",
        "matching_rule": MATCHING_RULE,
        "replacement_procedure": (
            "Original pilot IDs are kept in manifest order if eligible "
            "(content absent from the test split; first occurrence wins on "
            "within-sample duplicates). Freed slots are refilled with "
            "random.Random(seed).sample() over the sorted remaining "
            "eligible pool IDs."
        ),
        "pools": pools,
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot-size", type=int, default=20)
    ap.add_argument("--pilot", type=Path, default=ORIGINAL_PILOT_PATH,
                    help="original pilot manifest to derive from")
    ap.add_argument("--out", type=Path, default=DISJOINT_PILOT_PATH)
    args = ap.parse_args()

    original = json.loads(args.pilot.read_text(encoding="utf-8"))
    manifest = build_disjoint_pilot(PROCESSED_DIR, original,
                                    args.pilot_size, args.seed)
    args.out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for dataset, pool in manifest["pools"].items():
        print(
            f"{dataset}: {pool['selected_count']}/{args.pilot_size} selected; "
            f"{len(pool['kept_from_original_pilot'])} kept from original pilot, "
            f"{len(pool['excluded_from_original_pilot'])} excluded, "
            f"{len(pool['replacements'])} replacements "
            f"(pool of {pool['pool_size']}, "
            f"{pool['pool_items_whose_content_occurs_in_test']} items' content "
            "also in test split)"
        )
    print(f"wrote {args.out.relative_to(REPO_ROOT)}")
    if manifest["problems"]:
        for p in manifest["problems"]:
            print(f"WARNING: {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
