#!/usr/bin/env python3
"""Offline inspection of saved responses: raw text, reasoning slice, and
exact whitespace accounting.

Nothing here modifies the run -- it answers "did any processing step
lose blank lines / indentation / line endings?" for one record or for a
whole run directory.

Per-record view (--example-id + --condition):
  * raw_response printed verbatim between markers, plus its repr() so
    \\n / \\r / trailing spaces are unambiguous;
  * the stored reasoning_text and its repr();
  * whether the stored reasoning_text equals the exact substring of
    raw_response between the REASONING: and first FINAL_ANSWER: marker
    (recomputed live);
  * a line-numbered view marking <EMPTY> and <WS-ONLY> lines and
    trailing-whitespace counts;
  * whether the exported .txt file (if present under --export-root)
    preserves the raw response byte-for-byte in its RAW RESPONSE
    section.

Whole-run view (--scan): whitespace statistics per record.

Remember: JSONL stores newlines as the two-character escape \\n inside
the JSON string -- that is encoding, not data loss. This tool decodes
the JSON and inspects the real text.

Usage:
    python3 scripts/inspect_response.py --run-dir outputs/poetic_smoke_v3 \
        --example-id gsm8k-train-00204 --condition couplets
    python3 scripts/inspect_response.py --run-dir outputs/poetic_smoke_v3 --scan
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import common
import prompt_lib
import response_parsing as rp

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_records(run_dir: Path) -> list[dict]:
    src = run_dir / "responses.jsonl"
    if not src.exists():
        raise SystemExit(f"error: {src} not found")
    return common.read_jsonl(src)


def numbered_view(text: str) -> str:
    """Line-numbered rendering with explicit whitespace markers."""
    lines = text.split("\n")
    rows = []
    for i, ln in enumerate(lines, 1):
        if ln == "":
            body = "<EMPTY>"
        elif ln.strip() == "":
            body = f"<WS-ONLY {len(ln)} char(s): {ln!r}>"
        else:
            trailing = len(ln) - len(ln.rstrip(" \t"))
            cr = " <CR>" if ln.endswith("\r") else ""
            body = ln + (f"   <TRAILING-WS {trailing}>" if trailing else "") + cr
        rows.append(f"{i:>4}| {body}")
    return "\n".join(rows)


def reasoning_is_exact_slice(raw: str, reasoning: str | None,
                             expects_reasoning: bool) -> str:
    """Recompute extraction on the raw text and compare verbatim."""
    recomputed = rp.extract_reasoning_fields(raw, expects_reasoning)
    if recomputed["reasoning_text"] != reasoning:
        return ("MISMATCH: stored reasoning_text differs from a fresh "
                "extraction of raw_response")
    if reasoning is None:
        return "n/a (no reasoning_text stored)"
    start = raw.find("REASONING:")
    if start == -1:
        return "n/a (no REASONING: marker in raw)"
    seg_start = re.compile(r"^[ \t]*REASONING:[ \t]*", re.MULTILINE)
    m = seg_start.search(raw)
    if m and raw.startswith(reasoning, m.end()):
        return "exact substring of raw_response between markers: YES"
    if reasoning in raw:
        return "substring of raw_response: YES (marker boundary unusual)"
    return "NO -- not found inside raw_response"


def export_preserves_raw(run_dir: Path, rec: dict, raw: str,
                         export_root: Path) -> str:
    cfg = json.loads((run_dir / "run_config.json").read_text(
        encoding="utf-8"))
    compat = cfg.get("compatibility", {})
    run_id = cfg.get("run_name", run_dir.name)
    model = rec.get("model") or {
        "repo": compat.get("model_repo"),
        "revision": compat.get("model_revision"),
    }
    tag = common.model_tag(model, compat)
    path = (export_root / run_id / rec["dataset"] / rec["condition"]
            / f"{tag}__{common.safe_tag(rec['dataset'])}__"
              f"{common.safe_tag(rec['condition'])}__"
              f"{common.safe_tag(rec['example_id'])}.txt")
    if not path.exists():
        return f"export file not found: {path}"
    text = path.read_text(encoding="utf-8")
    marker = "=== RAW RESPONSE ===\n"
    if marker not in text:
        return f"export {path} has no RAW RESPONSE section"
    section = text.split(marker, 1)[1]
    # render_txt ends the file with exactly one '\n' after raw.
    if section.endswith("\n"):
        section = section[:-1]
    if section == raw:
        return f"export RAW RESPONSE section is byte-identical: {path}"
    return (f"MISMATCH: export {path} RAW RESPONSE section differs from "
            f"saved raw_response (len {len(section)} vs {len(raw)})")


def show_record(run_dir: Path, rec: dict, export_root: Path) -> int:
    raw = rp.record_raw_text(rec)
    reasoning = rec.get("reasoning_text")
    expects = prompt_lib.expects_reasoning(rec["condition"])
    print(f"record: {rec['example_id']} | {rec['dataset']} | "
          f"{rec['condition']} | run {rec.get('run_id')}")
    print(f"reasoning_parse_status: {rec.get('reasoning_parse_status')}")
    print(f"finish_reason: {rec.get('finish_reason')}  "
          f"hit_output_limit: {rec.get('hit_output_limit')}")
    stats = whitespace_stats(raw)
    print(f"whitespace: {stats}")
    print()
    print("=== RAW RESPONSE (verbatim) ===")
    print(raw)
    print("=== repr(raw_response) ===")
    print(repr(raw))
    print()
    if reasoning is not None:
        print("=== reasoning_text (verbatim) ===")
        print(reasoning)
        print("=== repr(reasoning_text) ===")
        print(repr(reasoning))
        print()
    else:
        print(f"=== reasoning_text: {reasoning!r} ===")
    print("reasoning-vs-raw check:",
          reasoning_is_exact_slice(raw, reasoning, expects))
    print()
    print("=== LINE-NUMBERED VIEW (diagnostic only) ===")
    print(numbered_view(raw))
    print()
    print("export check:",
          export_preserves_raw(run_dir, rec, raw, export_root))
    return 0


def whitespace_stats(raw: str) -> dict:
    lines = raw.split("\n")
    return {
        "chars": len(raw),
        "lines": len(lines),
        "empty_lines": sum(1 for l in lines if l == ""),
        "whitespace_only_lines": sum(
            1 for l in lines if l != "" and l.strip() == ""),
        "double_blank_sequences": raw.count("\n\n\n"),
        "cr_chars": raw.count("\r"),
        "trailing_space_lines": sum(
            1 for l in lines if l != l.rstrip(" \t")),
    }


def scan(run_dir: Path) -> int:
    recs = load_records(run_dir)
    print(f"{run_dir}: {len(recs)} record(s)")
    hdr = (f"  {'example_id':<28} {'condition':<18} {'empty':>5} "
           f"{'wsonly':>6} {'n3':>3} {'cr':>3} {'trail':>5}")
    print(hdr)
    for r in recs:
        s = whitespace_stats(rp.record_raw_text(r))
        print(f"  {r['example_id']:<28} {r['condition']:<18} "
              f"{s['empty_lines']:>5} {s['whitespace_only_lines']:>6} "
              f"{s['double_blank_sequences']:>3} "
              f"{s['cr_chars']:>3} {s['trailing_space_lines']:>5}")
    n_empty = sum(1 for r in recs
                  if any(l == "" for l in
                         rp.record_raw_text(r).split("\n")))
    n_triple = sum(1 for r in recs
                   if "\n\n\n" in rp.record_raw_text(r))
    n_cr = sum(1 for r in recs if "\r" in rp.record_raw_text(r))
    print(f"summary: {n_empty} raw responses contain >=1 empty line; "
          f"{n_triple} contain a double-blank (\\n\\n\\n) sequence; "
          f"{n_cr} contain CR/CRLF")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--example-id")
    ap.add_argument("--condition")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--export-root", type=Path,
                    default=REPO_ROOT / "exports")
    args = ap.parse_args()

    if args.scan:
        return scan(args.run_dir)

    if not (args.example_id and args.condition):
        ap.error("give --example-id and --condition, or --scan")
    recs = [r for r in load_records(args.run_dir)
            if r["example_id"] == args.example_id
            and r["condition"] == args.condition]
    if not recs:
        print(f"error: no record for {args.example_id} / "
              f"{args.condition} in {args.run_dir}", file=sys.stderr)
        return 1
    rc = 0
    for rec in recs:
        rc |= show_record(args.run_dir, rec, args.export_root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
