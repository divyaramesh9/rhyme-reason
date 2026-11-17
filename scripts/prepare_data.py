#!/usr/bin/env python3
"""Dataset preparation for the poetic-constraints reasoning study.

Prepares two datasets into a standardized JSONL layout:

  * GSM8K        (openai/grade-school-math, main train/test splits)
  * LogiQA v2    (csitfun/LogiQA2.0, English MRC train/dev/test splits;
                 "dev" is standardized to the name "validation")

For every prepared split the script writes two files under data/processed/:

  <dataset>_<split>.inputs.jsonl   -- model-visible fields only
  <dataset>_<split>.answers.jsonl  -- gold answers / reference reasoning

LogiQA additionally gets <dataset>_<split>.metadata.jsonl carrying the
source "type" reasoning-category annotations (not model inputs).

Usage:
    python3 scripts/prepare_data.py --download   # fetch + verify raw data, then prepare
    python3 scripts/prepare_data.py              # prepare from existing data/raw/
    python3 scripts/prepare_data.py --seed 7 --pilot-size 10

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

import common

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
REPORT_DIR = REPO_ROOT / "reports"
SOURCES_PATH = MANIFEST_DIR / "sources.json"
PILOT_PATH = MANIFEST_DIR / "pilot.json"
REPORT_PATH = REPORT_DIR / "data_validation.json"

# Pinned upstream revisions (immutable raw.githubusercontent URLs).
GSM8K_REPO = "https://github.com/openai/grade-school-math"
GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
LOGIQA_REPO = "https://github.com/csitfun/LogiQA2.0"
LOGIQA_COMMIT = "955e1d3df6c59d9bfb44d9913da1e1a27ec14e18"

DATASET_LICENSES = {
    "gsm8k": "MIT (see repository LICENSE)",
    "logiqa_v2": "CC BY-NC-SA 4.0 (see repository README)",
}

# Raw files required for this stage. `std_split` is the standardized split
# name used in all prepared outputs.
RAW_FILES = [
    {
        "dataset": "gsm8k",
        "raw_split": "train",
        "std_split": "train",
        "filename": "train.jsonl",
        "url": f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM8K_COMMIT}/grade_school_math/data/train.jsonl",
    },
    {
        "dataset": "gsm8k",
        "raw_split": "test",
        "std_split": "test",
        "filename": "test.jsonl",
        "url": f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM8K_COMMIT}/grade_school_math/data/test.jsonl",
    },
    {
        "dataset": "logiqa_v2",
        "raw_split": "train",
        "std_split": "train",
        "filename": "train.txt",
        "url": f"https://raw.githubusercontent.com/csitfun/LogiQA2.0/{LOGIQA_COMMIT}/logiqa/DATA/LOGIQA/train.txt",
    },
    {
        "dataset": "logiqa_v2",
        "raw_split": "dev",
        "std_split": "validation",
        "filename": "dev.txt",
        "url": f"https://raw.githubusercontent.com/csitfun/LogiQA2.0/{LOGIQA_COMMIT}/logiqa/DATA/LOGIQA/dev.txt",
    },
    {
        "dataset": "logiqa_v2",
        "raw_split": "test",
        "std_split": "test",
        "filename": "test.txt",
        "url": f"https://raw.githubusercontent.com/csitfun/LogiQA2.0/{LOGIQA_COMMIT}/logiqa/DATA/LOGIQA/test.txt",
    },
]

OPTION_LABELS = ("A", "B", "C", "D")
DATASETS = ("gsm8k", "logiqa_v2")
STD_SPLITS = ("train", "validation", "test")

# Fields that must never appear in model-input records.
ANSWER_ONLY_FIELDS = ("gold_answer", "source_answer", "reference_reasoning", "answer")

GSM8K_FINAL_DELIMITER = "####"
# Post-normalization accepted shapes: integer or decimal, optional minus sign.
NUMERIC_FINAL_RE = re.compile(r"^-?\d+(\.\d+)?$")


class MalformedRecordError(ValueError):
    """Raised when a source record cannot be standardized safely."""


def make_example_id(dataset: str, std_split: str, position: int) -> str:
    """Deterministic ID: dataset, standardized split, 0-based raw position."""
    return f"{dataset}-{std_split}-{position:05d}"


def read_jsonl(path: Path, errors: list[dict], tag: str) -> list[dict]:
    """Parse a JSON-lines file, tolerating a missing trailing newline."""
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(
                    {"file": tag, "line": lineno, "error": f"invalid JSON: {e}"}
                )
    return records


def dump_jsonl(records: list[dict]) -> str:
    """Deterministic serialization: insertion-ordered keys, UTF-8, LF endings."""
    return "".join(
        json.dumps(r, ensure_ascii=False, separators=(", ", ": ")) + "\n"
        for r in records
    )


def write_if_changed(path: Path, content: str) -> str:
    """Write content; return 'created', 'unchanged', or 'modified'.

    'modified' means a previously existing file's bytes differed -- the
    condition that would signal non-deterministic output on rerun.
    """
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == content:
        return "unchanged"
    path.write_text(content, encoding="utf-8")
    return "modified" if old is not None else "created"


# --------------------------------------------------------------------------
# GSM8K
# --------------------------------------------------------------------------

def extract_gsm8k_final(answer: str) -> tuple[str, str]:
    """Split a GSM8K answer into (reference_reasoning, normalized_final).

    Normalization rule: strip surrounding whitespace, then remove ','
    thousands separators. The result must match -?<digits>(.<digits>)? or the
    record is flagged as ambiguous rather than guessed.
    """
    if answer.count(GSM8K_FINAL_DELIMITER) != 1:
        raise MalformedRecordError(
            f"expected exactly one '{GSM8K_FINAL_DELIMITER}' delimiter, "
            f"found {answer.count(GSM8K_FINAL_DELIMITER)}"
        )
    reasoning, final = answer.rsplit(GSM8K_FINAL_DELIMITER, 1)
    final_raw = final.strip()
    normalized = final_raw.replace(",", "")
    if not NUMERIC_FINAL_RE.fullmatch(normalized):
        raise MalformedRecordError(
            f"final answer {final_raw!r} is not a plain number after "
            "removing thousands separators"
        )
    return reasoning.strip(), normalized


def transform_gsm8k(records: list[dict], std_split: str) -> tuple[list, list, list]:
    inputs, answers, errors = [], [], []
    for pos, rec in enumerate(records):
        eid = make_example_id("gsm8k", std_split, pos)
        try:
            if set(rec) != {"question", "answer"}:
                raise MalformedRecordError(
                    f"unexpected keys {sorted(rec)} (expected question/answer)"
                )
            if not isinstance(rec["question"], str) or not rec["question"].strip():
                raise MalformedRecordError("missing or empty 'question'")
            if not isinstance(rec["answer"], str) or not rec["answer"].strip():
                raise MalformedRecordError("missing or empty 'answer'")
            reasoning, gold = extract_gsm8k_final(rec["answer"])
        except MalformedRecordError as e:
            errors.append(
                {"example_id": eid, "dataset": "gsm8k", "split": std_split,
                 "position": pos, "error": str(e)}
            )
            continue
        inputs.append(
            {
                "example_id": eid,
                "dataset": "gsm8k",
                "split": std_split,
                "source_id": None,  # GSM8K provides no per-example identifier
                "question": rec["question"],
                "context": None,
                "options": None,
            }
        )
        answers.append(
            {
                "example_id": eid,
                "gold_answer": gold,
                "source_answer": rec["answer"],
                "reference_reasoning": reasoning,
            }
        )
    return inputs, answers, errors


# --------------------------------------------------------------------------
# LogiQA v2 (English MRC)
# --------------------------------------------------------------------------

def letter_from_index(index: int, n_options: int) -> str:
    """Map a 0-based source answer index to an option letter."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise MalformedRecordError(f"answer index {index!r} is not an integer")
    if not 0 <= index < n_options or index >= len(OPTION_LABELS):
        raise MalformedRecordError(
            f"answer index {index!r} out of range for {n_options} options"
        )
    return OPTION_LABELS[index]


def transform_logiqa(
    records: list[dict], std_split: str
) -> tuple[list, list, list, list]:
    inputs, answers, metadata, errors = [], [], [], []
    for pos, rec in enumerate(records):
        eid = make_example_id("logiqa_v2", std_split, pos)
        try:
            if not isinstance(rec.get("text"), str) or not rec["text"].strip():
                raise MalformedRecordError("missing or empty 'text' (passage)")
            if not isinstance(rec.get("question"), str) or not rec["question"].strip():
                raise MalformedRecordError("missing or empty 'question'")
            options = rec.get("options")
            if (
                not isinstance(options, list)
                or not options
                or len(options) > len(OPTION_LABELS)
            ):
                raise MalformedRecordError(
                    f"'options' must be a list of 1-{len(OPTION_LABELS)} strings"
                )
            if any(not isinstance(o, str) or not o.strip() for o in options):
                raise MalformedRecordError("empty option text")
            # Source convention verified against upstream README + code:
            # 'answer' is a 0-based index into 'options'.
            gold = letter_from_index(rec.get("answer"), len(options))
        except MalformedRecordError as e:
            errors.append(
                {"example_id": eid, "dataset": "logiqa_v2", "split": std_split,
                 "position": pos, "error": str(e)}
            )
            continue
        inputs.append(
            {
                "example_id": eid,
                "dataset": "logiqa_v2",
                "split": std_split,
                "source_id": rec.get("id"),
                "question": rec["question"],
                "context": rec["text"],
                # Option text is preserved verbatim; labels live in the keys
                # so existing letter-like prefixes are never duplicated.
                "options": dict(zip(OPTION_LABELS, options)),
            }
        )
        answers.append(
            {
                "example_id": eid,
                "gold_answer": gold,
                "source_answer": rec["answer"],  # original 0-based index
                "reference_reasoning": None,
            }
        )
        metadata.append(
            {
                "example_id": eid,
                "source_id": rec.get("id"),
                "type": rec.get("type"),
            }
        )
    return inputs, answers, metadata, errors


# --------------------------------------------------------------------------
# Download / integrity
# --------------------------------------------------------------------------

def load_sources() -> dict | None:
    if SOURCES_PATH.exists():
        return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    return None


def expected_checksums() -> dict[str, str]:
    """Map raw relative path -> expected sha256 from the sources manifest."""
    manifest = load_sources()
    expected = {}
    if manifest:
        for ds in manifest["datasets"].values():
            for f in ds["files"]:
                expected[f["local_path"]] = f["sha256"]
    return expected


def download_raw() -> None:
    """Fetch raw files, skipping ones already present, and write sources.json.

    If a sources manifest already exists, existing files are verified against
    its checksums and mismatches abort rather than overwriting the preserved
    originals.
    """
    known = expected_checksums()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    datasets_meta: dict[str, dict] = {}

    for spec in RAW_FILES:
        rel = f"data/raw/{spec['dataset']}/{spec['filename']}"
        dest = REPO_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            digest = common.sha256_file(dest)
            if rel in known and known[rel] != digest:
                raise SystemExit(
                    f"error: {rel} exists but sha256 {digest} does not match "
                    f"sources.json ({known[rel]}). Resolve manually; refusing "
                    "to overwrite preserved raw data."
                )
            print(f"  present  {rel} (sha256 {digest[:12]}...)")
        else:
            print(f"  fetching {spec['url']}")
            with urllib.request.urlopen(spec["url"]) as resp:
                dest.write_bytes(resp.read())
            digest = common.sha256_file(dest)
            if rel in known and known[rel] != digest:
                dest.unlink()
                raise SystemExit(
                    f"error: downloaded {rel} but sha256 {digest} does not "
                    f"match sources.json ({known[rel]})."
                )
            print(f"           sha256 {digest}")

        repo = GSM8K_REPO if spec["dataset"] == "gsm8k" else LOGIQA_REPO
        commit = GSM8K_COMMIT if spec["dataset"] == "gsm8k" else LOGIQA_COMMIT
        meta = datasets_meta.setdefault(
            spec["dataset"],
            {"repository": repo, "commit": commit,
             "license": DATASET_LICENSES[spec["dataset"]], "files": []},
        )
        meta["files"].append(
            {
                "raw_split": spec["raw_split"],
                "standardized_split": spec["std_split"],
                "url": spec["url"],
                "local_path": rel,
                "sha256": digest,
                "bytes": dest.stat().st_size,
                "downloaded_at": common.utc_now(),
            }
        )

    sources = {
        "description": "Pinned upstream sources and integrity checksums for raw data.",
        "datasets": datasets_meta,
    }
    SOURCES_PATH.write_text(
        json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  wrote    {SOURCES_PATH.relative_to(REPO_ROOT)}")


def verify_raw_checksums() -> dict:
    """Return check result comparing raw files to sources.json."""
    expected = expected_checksums()
    details = []
    ok = True
    for spec in RAW_FILES:
        rel = f"data/raw/{spec['dataset']}/{spec['filename']}"
        path = REPO_ROOT / rel
        if not path.exists():
            ok = False
            details.append({"file": rel, "status": "missing"})
            continue
        digest = common.sha256_file(path)
        if rel in expected:
            match = digest == expected[rel]
            ok &= match
            details.append(
                {"file": rel, "sha256": digest,
                 "status": "match" if match else "MISMATCH"}
            )
        else:
            details.append(
                {"file": rel, "sha256": digest, "status": "no recorded checksum"}
            )
    return {"status": "pass" if ok else "fail", "files": details}


# --------------------------------------------------------------------------
# Pilot selection
# --------------------------------------------------------------------------

def select_pilot_ids(pool_ids: list[str], n: int, seed: int) -> list[str]:
    """Sample n IDs from a sorted pool with a dedicated Random(seed).

    Each dataset pool is sampled with a fresh random.Random(seed), so the
    selection for one dataset never depends on the other's sample size.
    """
    if n > len(pool_ids):
        raise ValueError(f"pilot size {n} exceeds pool of {len(pool_ids)}")
    return sorted(random.Random(seed).sample(sorted(pool_ids), n))


def build_pilot(
    pools: dict[str, list[str]], sample_size: int, seed: int
) -> dict:
    selections = {}
    for dataset, ids in pools.items():
        selections[dataset] = select_pilot_ids(ids, sample_size, seed)
    return {
        "name": "pilot",
        "purpose": (
            "Shared pilot subset for prompt development and debugging. Every "
            "experimental condition must use exactly these example_ids; the "
            "underlying splits are unchanged."
        ),
        "seed": seed,
        "sample_size_per_dataset": sample_size,
        "selection": "random.Random(seed).sample(sorted(example_ids), n) per dataset",
        "pools": {
            "gsm8k": {
                "split": "train",
                "pool_size": len(pools["gsm8k"]),
                "example_ids": selections["gsm8k"],
            },
            "logiqa_v2": {
                "split": "validation",
                "pool_size": len(pools["logiqa_v2"]),
                "example_ids": selections["logiqa_v2"],
            },
        },
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def check_input_schema(inputs: list[dict]) -> dict:
    bad = []
    for r in inputs:
        problems = []
        if not isinstance(r.get("example_id"), str):
            problems.append("example_id not str")
        if r.get("dataset") not in DATASETS:
            problems.append(f"bad dataset {r.get('dataset')!r}")
        if r.get("split") not in STD_SPLITS:
            problems.append(f"bad split {r.get('split')!r}")
        if not isinstance(r.get("question"), str) or not r["question"].strip():
            problems.append("question missing/empty")
        if r.get("dataset") == "gsm8k":
            if r.get("context") is not None or r.get("options") is not None:
                problems.append("gsm8k context/options must be null")
        elif r.get("dataset") == "logiqa_v2":
            if not isinstance(r.get("context"), str) or not r["context"].strip():
                problems.append("logiqa context missing/empty")
            opts = r.get("options")
            if not isinstance(opts, dict) or set(opts) - set(OPTION_LABELS):
                problems.append("logiqa options must be dict keyed by A-D")
            elif any(not isinstance(t, str) or not t.strip() for t in opts.values()):
                problems.append("logiqa option text missing/empty")
        leaked = [k for k in ANSWER_ONLY_FIELDS if k in r]
        if leaked:
            problems.append(f"answer fields leaked into inputs: {leaked}")
        if problems:
            bad.append({"example_id": r.get("example_id"), "problems": problems})
    return {
        "status": "pass" if not bad else "fail",
        "records_checked": len(inputs),
        "violations": bad[:50],
        "violation_count": len(bad),
    }


def check_answer_schema(answers: list[dict]) -> dict:
    bad = []
    for r in answers:
        problems = []
        if not isinstance(r.get("example_id"), str):
            problems.append("example_id not str")
        if not isinstance(r.get("gold_answer"), str) or not r["gold_answer"]:
            problems.append("gold_answer missing/not str")
        if "source_answer" not in r:
            problems.append("source_answer missing")
        if not (r.get("reference_reasoning") is None or isinstance(r["reference_reasoning"], str)):
            problems.append("reference_reasoning must be str or null")
        if problems:
            bad.append({"example_id": r.get("example_id"), "problems": problems})
    return {
        "status": "pass" if not bad else "fail",
        "records_checked": len(answers),
        "violations": bad[:50],
        "violation_count": len(bad),
    }


def check_unique_ids(inputs: list[dict]) -> dict:
    seen, dups = set(), []
    for r in inputs:
        if r["example_id"] in seen:
            dups.append(r["example_id"])
        seen.add(r["example_id"])
    return {
        "status": "pass" if not dups else "fail",
        "total": len(inputs),
        "unique": len(seen),
        "duplicates": dups[:50],
    }


def check_alignment(inputs: list[dict], answers: list[dict]) -> dict:
    in_ids = [r["example_id"] for r in inputs]
    an_ids = [r["example_id"] for r in answers]
    return {
        "status": "pass" if in_ids == an_ids else "fail",
        "input_records": len(in_ids),
        "answer_records": len(an_ids),
        "same_order": in_ids == an_ids,
        "same_set": set(in_ids) == set(an_ids),
    }


def check_logiqa_mappings(inputs: list[dict], answers: list[dict]) -> dict:
    ans_by_id = {r["example_id"]: r for r in answers}
    bad = []
    checked = 0
    for r in inputs:
        if r["dataset"] != "logiqa_v2":
            continue
        checked += 1
        a = ans_by_id[r["example_id"]]
        opts = r["options"]
        problems = []
        if list(opts) != list(OPTION_LABELS)[: len(opts)]:
            problems.append(f"option labels/order wrong: {list(opts)}")
        idx = a["source_answer"]
        if a["gold_answer"] not in OPTION_LABELS:
            problems.append(f"gold_answer {a['gold_answer']!r} not A-D")
        elif isinstance(idx, int) and OPTION_LABELS[idx] != a["gold_answer"]:
            problems.append(f"index {idx} does not map to {a['gold_answer']}")
        if a["reference_reasoning"] is not None:
            problems.append("reference_reasoning must be null for logiqa")
        if problems:
            bad.append({"example_id": r["example_id"], "problems": problems})
    return {
        "status": "pass" if not bad else "fail",
        "records_checked": checked,
        "violations": bad[:50],
        "violation_count": len(bad),
    }


def check_gsm8k_extraction(answers: list[dict]) -> dict:
    bad, checked = [], 0
    for r in answers:
        if not r["example_id"].startswith("gsm8k-"):
            continue
        checked += 1
        problems = []
        if not NUMERIC_FINAL_RE.fullmatch(r["gold_answer"]):
            problems.append(f"gold_answer {r['gold_answer']!r} not normalized numeric")
        if GSM8K_FINAL_DELIMITER not in r["source_answer"]:
            problems.append("source_answer lost the #### delimiter (not preserved)")
        if not r["reference_reasoning"]:
            problems.append("reference_reasoning empty")
        if problems:
            bad.append({"example_id": r["example_id"], "problems": problems})
    return {
        "status": "pass" if not bad else "fail",
        "records_checked": checked,
        "violations": bad[:50],
        "violation_count": len(bad),
    }


def check_counts(source_counts: dict, written_counts: dict) -> dict:
    rows = []
    ok = True
    for key, src in sorted(source_counts.items()):
        got = written_counts.get(key, 0)
        match = src == got
        ok &= match
        rows.append({"split_key": key, "source_records": src,
                     "prepared_records": got, "match": match})
    return {"status": "pass" if ok else "fail", "splits": rows}


def check_duplicate_questions(inputs: list[dict]) -> dict:
    """Exact-duplicate detection within and across splits (report only)."""

    def key(r):
        if r["dataset"] == "logiqa_v2":
            return ("logiqa", r["context"], r["question"],
                    tuple(r["options"][k] for k in sorted(r["options"])))
        return ("gsm8k", r["question"])

    groups: dict[tuple, list[str]] = {}
    for r in inputs:
        groups.setdefault(key(r), []).append(r["example_id"])
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    cross = sum(1 for v in dups.values() if len({i.rsplit("-", 1)[0] for i in v}) > 1)
    per_dataset: dict[str, int] = {}
    for v in dups.values():
        for eid in v[1:]:  # extra copies beyond the first
            per_dataset[eid.split("-")[0]] = per_dataset.get(eid.split("-")[0], 0) + 1
    return {
        "status": "pass",  # informational: duplicates are kept, not deleted
        "duplicate_groups": len(dups),
        "cross_split_groups": cross,
        "extra_copies_per_dataset": per_dataset,
        "groups": [
            {"example_ids": v} for v in sorted(dups.values(), key=lambda x: x[0])
        ],
    }


def check_pilot(pilot: dict) -> dict:
    problems = []
    for dataset, info in pilot["pools"].items():
        split = info["split"]
        if split == "test":
            problems.append(f"{dataset}: pilot drawn from test split")
        for eid in info["example_ids"]:
            if not eid.startswith(f"{dataset}-{split}-"):
                problems.append(f"{eid} not in {dataset}/{split}")
        if len(info["example_ids"]) != len(set(info["example_ids"])):
            problems.append(f"{dataset}: duplicate pilot ids")
    return {
        "status": "pass" if not problems else "fail",
        "problems": problems,
        "seed": pilot["seed"],
        "counts": {d: len(i["example_ids"]) for d, i in pilot["pools"].items()},
    }


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def prepare(seed: int, pilot_size: int) -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at": common.utc_now(),
        "seed": seed,
        "pilot_size": pilot_size,
        "split_mapping": {
            "gsm8k": {"train": "train", "test": "test"},
            "logiqa_v2": {"train": "train", "dev": "validation", "test": "test"},
        },
        "checks": {},
        "unresolved_errors": [],
    }

    checks = report["checks"]
    checks["raw_checksums"] = verify_raw_checksums()
    if checks["raw_checksums"]["status"] != "pass":
        print("error: raw data missing or failed checksum verification. "
              "Run: python3 scripts/prepare_data.py --download", file=sys.stderr)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        return 1

    # ---- transform every raw file -------------------------------------
    source_counts, written_counts = {}, {}
    all_inputs, all_answers = [], []
    changed_flags = []

    for spec in RAW_FILES:
        rel = f"data/raw/{spec['dataset']}/{spec['filename']}"
        tag = f"{spec['dataset']}/{spec['raw_split']}"
        key = f"{spec['dataset']}_{spec['std_split']}"
        errors: list[dict] = []
        records = read_jsonl(REPO_ROOT / rel, errors, tag)
        source_counts[key] = len(records)

        if spec["dataset"] == "gsm8k":
            inputs, answers, terr = transform_gsm8k(records, spec["std_split"])
            metadata = None
        else:
            inputs, answers, metadata, terr = transform_logiqa(
                records, spec["std_split"]
            )
        errors.extend(terr)
        report["unresolved_errors"].extend(errors)
        if errors:
            continue  # do not write partial outputs for this split

        written_counts[key] = len(inputs)
        all_inputs.extend(inputs)
        all_answers.extend(answers)

        for suffix, recs in (("inputs", inputs), ("answers", answers),
                             ("metadata", metadata)):
            if recs is None:
                continue
            path = PROCESSED_DIR / f"{key}.{suffix}.jsonl"
            changed_flags.append((path.name, write_if_changed(path, dump_jsonl(recs))))

    if report["unresolved_errors"]:
        print(f"error: {len(report['unresolved_errors'])} malformed source "
              "record(s); see reports/data_validation.json", file=sys.stderr)
        checks["record_counts"] = check_counts(source_counts, written_counts)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        return 1

    # ---- pilot manifest ------------------------------------------------
    pools = {
        "gsm8k": [r["example_id"] for r in all_inputs
                  if r["dataset"] == "gsm8k" and r["split"] == "train"],
        "logiqa_v2": [r["example_id"] for r in all_inputs
                      if r["dataset"] == "logiqa_v2" and r["split"] == "validation"],
    }
    pilot = build_pilot(pools, pilot_size, seed)
    pilot_changed = write_if_changed(
        PILOT_PATH, json.dumps(pilot, indent=2, ensure_ascii=False) + "\n"
    )

    # ---- checks --------------------------------------------------------
    checks["input_schema"] = check_input_schema(all_inputs)
    checks["answer_schema"] = check_answer_schema(all_answers)
    checks["unique_example_ids"] = check_unique_ids(all_inputs)
    checks["input_answer_alignment"] = check_alignment(all_inputs, all_answers)
    checks["logiqa_option_mapping"] = check_logiqa_mappings(all_inputs, all_answers)
    checks["gsm8k_answer_extraction"] = check_gsm8k_extraction(all_answers)
    checks["record_counts"] = check_counts(source_counts, written_counts)
    checks["duplicate_questions"] = check_duplicate_questions(all_inputs)
    checks["pilot_integrity"] = check_pilot(pilot)

    # Re-running must not change previously written bytes or pilot IDs.
    modified = [n for n, c in changed_flags if c == "modified"]
    stable = not modified and pilot_changed != "modified"
    checks["deterministic_rerun"] = {
        "status": "pass" if stable else "fail",
        "note": "prepared files and pilot manifest identical to previous run"
        if stable else "outputs changed between runs",
        "files_modified": modified,
        "pilot_status": pilot_changed,
    }

    n_fail = sum(1 for c in checks.values() if c["status"] == "fail")
    report["status"] = "pass" if n_fail == 0 else "fail"
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"prepared {len(all_inputs)} examples across "
          f"{len(written_counts)} splits")
    for key in sorted(written_counts):
        print(f"  {key}: {written_counts[key]}")
    print(f"pilot: {pilot_size} gsm8k-train + {pilot_size} logiqa_v2-validation "
          f"(seed {seed}) -> data/manifests/pilot.json")
    print(f"validation: {report['status']} "
          f"({len(checks) - n_fail}/{len(checks)} checks passed) "
          f"-> reports/data_validation.json")
    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="fetch raw files (checksum-verified) before preparing")
    ap.add_argument("--seed", type=int, default=42,
                    help="pilot selection seed (default: 42)")
    ap.add_argument("--pilot-size", type=int, default=20,
                    help="pilot examples per dataset (default: 20)")
    args = ap.parse_args()

    if args.download:
        print("downloading raw data (pinned commits)...")
        download_raw()
    return prepare(args.seed, args.pilot_size)


if __name__ == "__main__":
    sys.exit(main())
