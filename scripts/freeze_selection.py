#!/usr/bin/env python3
"""Validate and freeze the manual device selection for the final
cross-model comparison.

Workflow:

  1. Edit configs/device_selection.json -- one "retain"/"exclude"
     decision + reason per device, based on YOUR screening review.
  2. python3 scripts/freeze_selection.py freeze
       validates the file (all 8 devices decided, reasons non-empty),
       then writes configs/device_selection.frozen.json recording the
       retained set, the sha256 of every device prompt template and the
       shared system prompt, the source runs/review files, and the
       selection file's hash. A frozen set with ZERO retained devices
       is reported loudly and refused unless --allow-empty is passed --
       the tool never loosens the criteria for you.
  3. python3 scripts/freeze_selection.py verify
       re-hashes the prompt files and fails if any drifted since
       freezing -- run this before final generation to prevent
       evaluating with silently changed prompts.

Nothing here picks devices for you and there is no automatic
threshold: the retained set is exactly what you recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTION_PATH = REPO_ROOT / "configs" / "device_selection.json"
FROZEN_PATH = REPO_ROOT / "configs" / "device_selection.frozen.json"
PROMPTS_DIR = REPO_ROOT / "prompts"


def load_selection(path: Path = SELECTION_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_selection(sel: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (retained, excluded, problems)."""
    devices = sel.get("devices", {})
    problems, retained, excluded = [], [], []
    for name in study_defs.POETIC_DEVICES:
        d = devices.get(name)
        if d is None:
            problems.append(f"device {name!r} missing from selection file")
            continue
        decision = d.get("decision")
        if decision not in ("retain", "exclude"):
            problems.append(
                f"device {name!r}: decision must be 'retain' or "
                f"'exclude' (got {decision!r})")
            continue
        if not (d.get("reason") or "").strip():
            problems.append(f"device {name!r}: reason is empty")
        (retained if decision == "retain" else excluded).append(name)
    extra = set(devices) - set(study_defs.POETIC_DEVICES)
    if extra:
        problems.append(f"unknown device(s) in file: {sorted(extra)}")
    return retained, excluded, problems


def freeze(selection_path: Path, frozen_path: Path,
           allow_empty: bool) -> int:
    sel = load_selection(selection_path)
    retained, excluded, problems = validate_selection(sel)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1
    if not retained and not allow_empty:
        print("error: selection retains ZERO devices. The final "
              "comparison would have no poetic conditions. If you "
              "really mean this, pass --allow-empty; nothing was "
              "loosened automatically.", file=sys.stderr)
        return 1

    prompt_shas = {"system": common.sha256_file(PROMPTS_DIR / "system.txt")}
    for name in study_defs.POETIC_DEVICES + study_defs.CONTROLS:
        prompt_shas[name] = common.sha256_file(PROMPTS_DIR / f"{name}.txt")

    frozen = {
        "frozen_at": common.utc_now(),
        "selection_file": str(selection_path),
        "selection_file_sha256": common.sha256_file(selection_path),
        "retained_devices": retained,
        "excluded_devices": excluded,
        "final_conditions": study_defs.CONTROLS + retained,
        "prompt_template_sha256": prompt_shas,
        "model_dataset_differences": sel.get(
            "model_dataset_differences", ""),
        "source_screening_runs": sel.get("source_screening_runs", []),
        "source_review_files": sel.get("source_review_files", []),
        "unresolved_judgments": sel.get("unresolved_judgments", ""),
        "limitations": sel.get("limitations", ""),
        "note": ("Common retained device set used for ALL models in "
                 "the final comparison; per-model differences are "
                 "documented descriptively, not substituted."),
    }
    frozen_path.write_text(json.dumps(frozen, indent=2,
                                      ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"froze {len(retained)} retained device(s): {retained}")
    print(f"excluded: {excluded}")
    print(f"final conditions: {frozen['final_conditions']}")
    if not retained:
        print("WARNING: frozen selection retains ZERO devices "
              "(--allow-empty)")
    print(f"wrote {frozen_path}")
    print("run 'verify' before final generation to confirm prompts "
          "have not drifted.")
    return 0


def verify(frozen_path: Path) -> int:
    if not frozen_path.exists():
        print("error: no frozen selection; run 'freeze' first",
              file=sys.stderr)
        return 1
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    problems = []
    for name, sha in frozen["prompt_template_sha256"].items():
        p = PROMPTS_DIR / f"{name}.txt"
        if not p.exists():
            problems.append(f"prompt file {p} missing")
            continue
        cur = common.sha256_file(p)
        if cur != sha:
            problems.append(
                f"prompt {name!r} changed since freeze "
                f"({sha[:12]}... -> {cur[:12]}...)")
    sel_path = Path(frozen["selection_file"])
    if sel_path.exists() and \
            common.sha256_file(sel_path) != frozen["selection_file_sha256"]:
        problems.append(
            f"selection file {sel_path} changed since freeze; "
            "re-freeze if the change is intended")
    for p in problems:
        print(f"DRIFT: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"frozen selection intact: "
          f"{frozen['retained_devices']} + controls "
          f"{study_defs.CONTROLS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--selection", type=Path, default=SELECTION_PATH)
    f.add_argument("--out", type=Path, default=FROZEN_PATH)
    f.add_argument("--allow-empty", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--frozen", type=Path, default=FROZEN_PATH)
    args = ap.parse_args()

    if args.cmd == "freeze":
        return freeze(args.selection, args.out, args.allow_empty)
    return verify(args.frozen)


if __name__ == "__main__":
    sys.exit(main())
