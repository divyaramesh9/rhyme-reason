#!/usr/bin/env python3
"""Build (or check) the matched development-pilot manifest.

Reads configs/dev_pilot.json, verifies its device list against the
FROZEN selection (mismatches are reported, never auto-fixed), then
selects ``questions_per_dataset`` examples per dataset:

  * from the development pools (gsm8k train, logiqa_v2 validation);
  * excluding every example_id already used in screening / prompt
    development (saved run outputs + earlier manifests);
  * excluding items whose content occurs in the official test split
    (content-key matching, same rule as build_eval_manifest.py);
  * with no duplicated content inside the selection.

Everything is deterministic given the recorded seed; the manifest
stores the seed, the exclusions, and per-dataset provenance.

``--check`` derives all experiment counts from the config + manifest
and re-verifies disjointness and determinism. Setting
questions_per_dataset to 4 must report 8 questions / 192 responses /
120 poetic evaluations.

Usage:
    python3 scripts/build_dev_pilot.py            # build the manifest
    python3 scripts/build_dev_pilot.py --check    # verify + derive counts
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import common
import select_disjoint_pilot as sdp
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "dev_pilot.json"
FROZEN_PATH = REPO_ROOT / "configs" / "device_selection.frozen.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PRIOR_MANIFESTS = ["pilot.json", "pilot_disjoint.json", "screening4.json"]

POOLS = {
    "gsm8k": {"pool_split": "train", "test_split": "test"},
    "logiqa_v2": {"pool_split": "validation", "test_split": "test"},
}
DATASET_ORDER = ("gsm8k", "logiqa_v2")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_frozen(config: dict, frozen_path: Path = FROZEN_PATH) -> list[str]:
    """Compare the config's device list to the frozen selection."""
    problems = []
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    retained = sorted(frozen["retained_devices"])
    configured = sorted(config["devices"])
    if configured != retained:
        problems.append(
            f"MISMATCH: dev_pilot devices {configured} != frozen "
            f"retained devices {retained} -- fix the config or the "
            "frozen file; nothing was changed automatically")
    if sorted(config["controls"]) != sorted(study_defs.CONTROLS):
        problems.append(
            f"MISMATCH: controls {config['controls']} != "
            f"{study_defs.CONTROLS}")
    return problems


def used_example_ids(outputs_dir: Path,
                     manifest_dir: Path) -> tuple[set, dict]:
    """All example_ids already used in saved runs or earlier manifests,
    plus a provenance breakdown."""
    used, sources = set(), {}

    def _disp(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    for kind in ("responses.jsonl", "failures.jsonl"):
        for p in sorted(outputs_dir.glob(f"*/{kind}")):
            ids = {r["example_id"] for r in common.read_jsonl(p)}
            sources[_disp(p)] = len(ids)
            used |= ids
    for name in PRIOR_MANIFESTS:
        p = manifest_dir / name
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        ids = {eid for pool in m.get("pools", {}).values()
               for eid in pool.get("example_ids", [])}
        sources[_disp(p)] = len(ids)
        used |= ids
    return used, sources


def select_pool(pool_records: dict, test_keys: set, used: set,
                n: int, seed: int) -> dict:
    """Deterministic sample of n eligible ids."""
    seen_keys, eligible = set(), []
    for eid in sorted(pool_records):
        if eid in used:
            continue
        k = sdp.content_key(pool_records[eid])
        if k in test_keys or k in seen_keys:
            continue
        seen_keys.add(k)
        eligible.append(eid)
    chosen = sorted(random.Random(seed).sample(
        eligible, min(n, len(eligible))))
    return {"example_ids": chosen, "eligible_count": len(eligible),
            "shortfall": max(0, n - len(chosen))}


def build_manifest(config: dict, config_path: Path,
                   processed_dir: Path,
                   outputs_dir: Path, manifest_dir: Path) -> dict:
    n = config["questions_per_dataset"]
    used, sources = used_example_ids(outputs_dir, manifest_dir)
    pools, problems = {}, []
    for dataset in DATASET_ORDER:
        spec = POOLS[dataset]
        pool = {r["example_id"]: r for r in common.read_jsonl(
            processed_dir / f"{dataset}_{spec['pool_split']}.inputs.jsonl")}
        test_keys = {sdp.content_key(r) for r in common.read_jsonl(
            processed_dir / f"{dataset}_{spec['test_split']}.inputs.jsonl")}
        sel = select_pool(pool, test_keys, used, n, config["seed"])
        if sel["shortfall"]:
            problems.append(
                f"{dataset}: only {len(sel['example_ids'])} eligible "
                f"example(s) remain after exclusions; requested {n}. "
                "Refusing to shrink the sample or reuse test items -- "
                "resolve manually.")
        pools[dataset] = {
            "split": spec["pool_split"],
            "pool_size": len(pool),
            "excluded_used_in_development":
                len(set(pool) & used),
            "excluded_content_in_test_split": sum(
                1 for r in pool.values()
                if sdp.content_key(r) in test_keys),
            "eligible_count": sel["eligible_count"],
            "example_ids": sel["example_ids"],
            "selected_count": len(sel["example_ids"]),
        }
    return {
        "name": config["name"],
        "stage": config["generation"]["stage"],
        "purpose": ("Matched development pilot: identical questions "
                    "across every model and condition; disjoint from "
                    "screening/prompt-development items and from the "
                    "official test split (content-level)."),
        "seed": config["seed"],
        "questions_per_dataset": n,
        "config": str(config_path.relative_to(REPO_ROOT))
        if config_path.is_relative_to(REPO_ROOT) else str(config_path),
        "matching_rule": sdp.MATCHING_RULE,
        "selection_procedure": (
            "Exclude IDs already used in saved runs or earlier "
            "manifests, then items whose content key occurs in the test "
            "split, then content duplicates; draw "
            "random.Random(seed).sample over the sorted eligible IDs."),
        "exclusion_sources": sources,
        "conditions": config["controls"] + config["devices"],
        "models": config["models"],
        "pools": pools,
        "problems": problems,
    }


def derive_counts(config: dict, manifest: dict) -> dict:
    """All experiment counts derived from config + manifest."""
    n_q = sum(len(p["example_ids"]) for p in manifest["pools"].values())
    n_models = len(config["models"])
    n_dev, n_ctrl = len(config["devices"]), len(config["controls"])
    return {
        "total_questions": n_q,
        "questions_per_dataset": config["questions_per_dataset"],
        "models": n_models,
        "conditions_total": n_dev + n_ctrl,
        "generation_responses": n_q * (n_dev + n_ctrl) * n_models,
        "poetic_evaluations": n_q * n_dev * n_models,
        "control_responses": n_q * n_ctrl * n_models,
    }


def check(config: dict, manifest_path: Path, processed_dir: Path) -> int:
    problems = check_frozen(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    c = derive_counts(config, manifest)
    print("derived counts:")
    for k, v in c.items():
        print(f"  {k}: {v}")

    # Live disjointness re-verification.
    for dataset, pool in manifest["pools"].items():
        spec = POOLS[dataset]
        inputs = {r["example_id"]: r for r in common.read_jsonl(
            processed_dir / f"{dataset}_{spec['pool_split']}.inputs.jsonl")}
        test_keys = {sdp.content_key(r) for r in common.read_jsonl(
            processed_dir / f"{dataset}_{spec['test_split']}.inputs.jsonl")}
        keys = set()
        for eid in pool["example_ids"]:
            rec = inputs.get(eid)
            if rec is None:
                problems.append(f"{eid} missing from {dataset} "
                                f"{spec['pool_split']} inputs")
                continue
            k = sdp.content_key(rec)
            if k in test_keys:
                problems.append(f"{eid}: content occurs in test split")
            if k in keys:
                problems.append(f"{eid}: duplicate content in sample")
            keys.add(k)

    # Determinism: rebuilding must reproduce the same IDs.
    rebuilt = build_manifest(config, manifest_path,
                             processed_dir,
                             OUTPUTS_DIR, manifest_path.parent)
    for dataset in DATASET_ORDER:
        if (rebuilt["pools"][dataset]["example_ids"]
                != manifest["pools"][dataset]["example_ids"]):
            problems.append(
                f"{dataset}: manifest is not reproducible from seed "
                f"{config['seed']} -- selections differ")

    if manifest.get("problems"):
        problems += manifest["problems"]
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"manifest OK: {manifest_path} "
          f"(deterministic, test-disjoint, frozen-selection match)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="output path (default: config's 'manifest')")
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--check", action="store_true",
                    help="verify manifest + derive counts; writes nothing")
    args = ap.parse_args()

    config = load_config(args.config)
    manifest_path = args.manifest or REPO_ROOT / config["manifest"]

    if args.check:
        if not manifest_path.exists():
            print(f"error: {manifest_path} not built yet",
                  file=sys.stderr)
            return 1
        return check(config, manifest_path, args.processed_dir)

    problems = check_frozen(config)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1
    manifest = build_manifest(config, args.config, args.processed_dir,
                              OUTPUTS_DIR, manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2,
                                        ensure_ascii=False) + "\n",
                             encoding="utf-8")
    c = derive_counts(config, manifest)
    for dataset, pool in manifest["pools"].items():
        print(f"{dataset}: {pool['selected_count']} selected of "
              f"{pool['eligible_count']} eligible "
              f"({pool['excluded_used_in_development']} used in dev, "
              f"{pool['excluded_content_in_test_split']} with test "
              f"content; pool {pool['pool_size']})")
        print(f"  ids: {pool['example_ids']}")
    print(f"derived counts: {c['total_questions']} questions, "
          f"{c['generation_responses']} responses "
          f"({c['poetic_evaluations']} poetic + "
          f"{c['control_responses']} control)")
    print(f"wrote {manifest_path}")
    if manifest["problems"]:
        for p in manifest["problems"]:
            print(f"WARNING: {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
