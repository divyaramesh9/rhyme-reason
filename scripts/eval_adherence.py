#!/usr/bin/env python3
"""Automated adherence evaluation over saved generation responses.

Subcommands:

  run            -- deterministic evaluation only (offline). Evaluates the
                    five frozen device conditions in the given run dirs and
                    appends derived records to <run>/adherence_evals.jsonl.
  status         -- coverage/verdict summary of existing eval records
                    (offline).
  judge-preview  -- show which responses the routing policy would send to
                    the LLM judge, one full example request, and an
                    offline cost estimate (no credentials, no writes).
  judge-run      -- execute the judge calls (PAID; sequential, ledger-
                    guarded). You run this; the agent does not.

Eval records are derived artifacts linked to immutable responses by
``eval_key`` (response id + content hash + rule/rubric/evaluator/judge
versions). Reruns skip keys already present, so rubric or judge changes
create new records rather than overwriting old judgments.

Usage:
    python3 scripts/eval_adherence.py run --run-dirs outputs/screening_qwen
    python3 scripts/eval_adherence.py status --run-dirs outputs/screening_*
    python3 scripts/eval_adherence.py judge-preview --run-dirs outputs/screening_*
    python3 scripts/eval_adherence.py judge-run --run-dirs outputs/screening_* \
        --judge-config configs/judge_gpt-5-mini-2025-08-07.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import adherence as ad
import backends
import budget as budget_mod
import common
import judge_client as jc
import phonetics as ph
import prepare_review as pr
import response_parsing as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ad.PROMPTS_DIR
RUBRIC_DIR = ad.RUBRIC_DIR
POLICY_PATH = REPO_ROOT / "configs" / "evaluator_policy.json"
DEFAULT_BUDGET_CONFIG = REPO_ROOT / "configs" / "budget.json"
EVALS_FILE = "adherence_evals.jsonl"

PURPOSE_VALIDATION = "judge:validation"
PURPOSE_PILOT = "judge:pilot"


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def load_policy(path: Path = POLICY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_sha(path: Path = POLICY_PATH) -> str:
    return common.sha256_file(path)


def run_meta(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "run_config.json").read_text(
        encoding="utf-8"))
    compat = cfg.get("compatibility", {})
    return {
        "run_id": cfg.get("run_name", run_dir.name),
        "compat": compat,
        "fallback_model": {
            "repo": compat.get("model_repo"),
            "revision": compat.get("model_revision"),
        },
    }


def iter_evaluatable(run_dir: Path, conditions=None):
    """Yield (response_id, record) for records whose condition has a
    deterministic evaluator (the five frozen devices)."""
    meta = run_meta(run_dir)
    for rec in common.read_jsonl(run_dir / "responses.jsonl"):
        if rec["condition"] not in ad.DEVICE_EVALUATORS:
            continue
        if conditions and rec["condition"] not in conditions:
            continue
        model = rec.get("model") or meta["fallback_model"]
        rid = pr.response_id(meta["run_id"], model, meta["compat"], rec)
        yield rid, rec, meta


def existing_keys(evals_path: Path) -> set[str]:
    return {r["eval_key"] for r in common.read_jsonl(evals_path)
            if "eval_key" in r}


def load_evals(run_dir: Path) -> list[dict]:
    return common.read_jsonl(run_dir / EVALS_FILE)


def _route_to_judge(policy: dict, device: str, det: dict) -> bool:
    """Does the routing policy send this deterministic result to the
    judge?"""
    mode = policy["devices"][device]["mode"]
    if mode == "deterministic":
        if device == "alliteration" and policy["devices"][device] \
                .get("judge_review") == "on_letter_sound_conflict":
            return any(
                (u.get("phonetic_evidence") or {}).get("agreement")
                is False
                for u in det.get("units", []))
        return False
    if mode == "deterministic_then_judge_unresolved":
        return det["verdict"] == "uncertain" or det["needs_judge"]
    if mode == "judge_always":
        return True
    raise ValueError(f"unknown policy mode {mode!r}")


def _evidence_text(det: dict, lexicon) -> str:
    """Compact mechanical evidence block for the judge."""
    lines = [f"deterministic verdict: {det['verdict']} "
             f"({det['processing_status']})"]
    for u in det.get("units", []):
        lines.append(f"- {u['detail']}")
        pe = u.get("phonetic_evidence")
        if pe:
            lines.append(f"    phonetic evidence: {pe['detail']}")
    if det.get("evidence_summary"):
        lines.append(f"notes: {det['evidence_summary']}")
    lines.append(f"lexicon: {ph.lexicon_status(lexicon)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# run -- deterministic evaluation
# --------------------------------------------------------------------------

def cmd_run(args) -> int:
    policy = load_policy()
    psha = policy_sha()
    lexicon = ph.load_lexicon(args.lexicon)
    lex_fp = ph.lexicon_fingerprint(lexicon)
    print(f"lexicon: {ph.lexicon_status(lexicon)}")
    n_new = n_skip = 0
    for run_dir in args.run_dirs:
        evals_path = run_dir / EVALS_FILE
        done = existing_keys(evals_path)
        wrote = 0
        for rid, rec, meta in iter_evaluatable(
                run_dir, args.conditions):
            key = ad.eval_key(rid, rec, "heuristic",
                              ad.rubric_sha256(rec["condition"]),
                              psha, None, lex_fp)
            if key in done:
                n_skip += 1
                continue
            det = ad.evaluate(rec, lexicon)
            det["needs_judge"] = _route_to_judge(
                policy, rec["condition"], det)
            record = {
                "eval_key": key,
                "response_id": rid,
                "run_id": meta["run_id"],
                "dataset": rec["dataset"],
                "example_id": rec["example_id"],
                "condition": rec["condition"],
                "source": "heuristic",
                "evaluator_version": ad.EVALUATOR_VERSION,
                "rule_sha256": ad.rule_sha256(rec["condition"]),
                "rubric_sha256": ad.rubric_sha256(rec["condition"]),
                "policy_sha256": psha,
                "response_sha256": common.sha256_text(
                    rp.record_raw_text(rec)),
                "judge": None,
                "recorded_at": common.utc_now(),
                **det,
            }
            common.append_jsonl(evals_path, record)
            done.add(key)
            wrote += 1
        n_new += wrote
        print(f"{run_dir}: +{wrote} deterministic eval record(s) "
              f"-> {evals_path}")
    print(f"total: {n_new} new, {n_skip} already cached (skipped)")
    return 0


# --------------------------------------------------------------------------
# status -- coverage summary
# --------------------------------------------------------------------------

def cmd_status(args) -> int:
    for run_dir in args.run_dirs:
        evals = load_evals(run_dir)
        responses = [r for r in common.read_jsonl(
            run_dir / "responses.jsonl")
            if r["condition"] in ad.DEVICE_EVALUATORS]
        print(f"\n=== {run_dir} ===")
        print(f"  poetic responses: {len(responses)}; "
              f"eval records: {len(evals)}")
        by: dict[tuple, dict] = {}
        for e in evals:
            cell = (e["condition"], e["source"])
            c = by.setdefault(cell, {"pass": 0, "fail": 0, "uncertain": 0,
                                     "not_evaluable": 0, "other": 0})
            c[e["verdict"] if e["verdict"] in c else "other"] += 1
        for (cond, src), c in sorted(by.items()):
            print(f"  {cond:<14} {src:<14} "
                  + " ".join(f"{k}={v}" for k, v in c.items()))
        pend = [e for e in evals if e.get("needs_judge")
                and e["source"] == "heuristic"]
        judged = {e["response_id"] for e in evals
                  if e["source"] == "llm_judge"}
        still = [e for e in pend if e["response_id"] not in judged]
        if still:
            print(f"  judge-pending: {len(still)} response(s) routed to "
                  "the judge but not yet judged")
    return 0


# --------------------------------------------------------------------------
# judge helpers
# --------------------------------------------------------------------------

def _judge_plan(args):
    """Yield (run_dir, response_id, rec, det_record, meta) needing a
    judge call under the routing policy."""
    for run_dir in args.run_dirs:
        heur = {}
        judged = set()
        for e in load_evals(run_dir):
            if e["source"] == "heuristic":
                heur[e["response_id"]] = e
            elif e["source"] == "llm_judge" and e.get(
                    "processing_status") != "judge_request_error":
                # Definitive (unbilled) request errors are retried;
                # every other llm_judge record counts as judged.
                judged.add(e["response_id"])
        for rid, rec, meta in iter_evaluatable(run_dir, args.conditions):
            det = heur.get(rid)
            if det is None:
                continue
            if rid in judged:
                continue
            if not det.get("needs_judge"):
                continue
            yield run_dir, rid, rec, det, meta


def _build_request(rec: dict, det: dict, lexicon,
                   judge_dir: Path) -> tuple[list[dict], str, str]:
    rule = (PROMPTS_DIR / f"{rec['condition']}.txt").read_text(
        encoding="utf-8")
    rubric_path = judge_dir / "rubrics" / f"{rec['condition']}.txt"
    rubric = rubric_path.read_text(encoding="utf-8")
    view, _ = ad.reasoning_view(rec)
    evidence = _evidence_text(det, lexicon)
    return jc.build_judge_messages(rule, rubric, view or "",
                                   evidence, judge_dir), view or "", \
        rubric


def cmd_judge_preview(args) -> int:
    judge_cfg = backends.load_model_config(args.judge_config)
    lexicon = ph.load_lexicon(args.lexicon)
    print(f"lexicon: {ph.lexicon_status(lexicon)}")
    plan = list(_judge_plan(args))
    print(f"judge model: {judge_cfg['model']} "
          f"(config {args.judge_config.name})")
    print(f"responses routed to the judge: {len(plan)}")
    total_in = 0
    first = None
    for run_dir, rid, rec, det, meta in plan:
        msgs, view, _ = _build_request(rec, det, lexicon, args.judge_dir)
        est = jc.JudgeClient(judge_cfg, client=object()) \
            .estimate_input_tokens(msgs)
        total_in += est
        if first is None:
            first = (rid, msgs, est)
    if not plan:
        print("nothing to judge")
        return 0
    p = judge_cfg["pricing"]
    max_out = judge_cfg.get("max_output_tokens", 600)
    conservative = (total_in * p["input_usd_per_1m"] / 1e6
                    + len(plan) * max_out
                    * p["output_usd_per_1m"] / 1e6)
    print(f"approx input tokens total: ~{total_in} "
          "(conservative chars/3 heuristic)")
    print(f"conservative cost estimate: ${conservative:.4f} "
          f"({len(plan)} requests x up to {max_out} output tokens, "
          "full price, no discounts)")
    print(f"requests exceeding ~{args.max_input_tokens_est} estimated "
          "input tokens are flagged 'judge_skipped_too_long', never "
          "truncated.")
    rid, msgs, est = first
    print(f"\n=== EXAMPLE REQUEST ({rid}, ~{est} input tokens) ===")
    for m in msgs:
        print(f"--- {m['role']} ---\n{m['content']}\n")
    print("PREVIEW ONLY -- no credentials needed, nothing sent or "
          "written.")
    return 0


def cmd_judge_run(args) -> int:
    judge_cfg = backends.load_model_config(args.judge_config)
    psha = policy_sha()
    lexicon = ph.load_lexicon(args.lexicon)
    lex_fp = ph.lexicon_fingerprint(lexicon)
    print(f"lexicon: {ph.lexicon_status(lexicon)}")
    plan = list(_judge_plan(args))
    if not plan:
        print("nothing to judge")
        return 0
    print(f"{len(plan)} response(s) routed to the judge "
          f"(model {judge_cfg['model']})")

    client = jc.JudgeClient(judge_cfg)
    client.prepare()
    ledger = budget_mod.BudgetLedger(args.budget_config, args.ledger)
    judge_cfg_sha = common.sha256_file(args.judge_config)
    max_out = judge_cfg.get("max_output_tokens", 600)
    purpose = args.purpose

    n_done = n_err = 0
    for i, (run_dir, rid, rec, det, meta) in enumerate(plan, 1):
        evals_path = run_dir / EVALS_FILE
        done = existing_keys(evals_path)
        msgs, view, _ = _build_request(rec, det, lexicon, args.judge_dir)
        est_in = client.estimate_input_tokens(msgs)

        base = {
            "response_id": rid, "run_id": meta["run_id"],
            "dataset": rec["dataset"], "example_id": rec["example_id"],
            "condition": rec["condition"], "source": "llm_judge",
            "evaluator_version": ad.EVALUATOR_VERSION,
            "rule_sha256": ad.rule_sha256(rec["condition"]),
            "rubric_sha256": ad.rubric_sha256(rec["condition"]),
            "policy_sha256": psha,
            "response_sha256": common.sha256_text(
                rp.record_raw_text(rec)),
        }

        if est_in > args.max_input_tokens_est:
            key = ad.eval_key(rid, rec, "llm_judge",
                              base["rubric_sha256"], psha,
                              judge_cfg["model"], lex_fp)
            if key not in done:
                common.append_jsonl(evals_path, {
                    **base, "eval_key": key, "verdict": "uncertain",
                    "processing_status": "judge_skipped_too_long",
                    "judge": {"model": judge_cfg["model"],
                              "est_input_tokens": est_in,
                              "messages": msgs},
                    "units": [], "evidence_summary":
                        "reasoning view exceeds judge input budget; "
                        "flagged, not truncated",
                    "recorded_at": common.utc_now()})
                print(f"  [{i}/{len(plan)}] {rid}: SKIPPED too long "
                      f"(~{est_in} est input tokens)")
            continue

        reservation = None
        try:
            est_cost = budget_mod.conservative_cost(
                judge_cfg, est_in, max_out)
            reservation = ledger.reserve(
                stage="judging", run_id=purpose,
                model=judge_cfg["model"], example_id=rec["example_id"],
                condition=rec["condition"], est_cost_usd=est_cost,
                est_input_tokens=est_in, max_output_tokens=max_out)
            result = client.judge(msgs)
        except budget_mod.BudgetExceeded as e:
            print(f"\nBUDGET STOP before [{i}/{len(plan)}] {rid}: {e}\n"
                  "completed judgments are saved; rerun to resume.")
            return 3
        except Exception as e:
            kind = backends.classify_provider_error(e)
            if reservation is not None:
                if kind == "ambiguous":
                    ledger.mark_unresolved(
                        reservation["reservation_id"],
                        run_id=purpose, example_id=rec["example_id"],
                        condition=rec["condition"], error=str(e))
                else:
                    ledger.void(reservation["reservation_id"],
                                reason="definitive_provider_error")
            key = ad.eval_key(rid, rec, "llm_judge",
                              base["rubric_sha256"], psha,
                              judge_cfg["model"], lex_fp)
            if key not in done:
                common.append_jsonl(evals_path, {
                    **base, "eval_key": key,
                    "verdict": "uncertain",
                    "processing_status": ("judge_unresolved"
                                          if kind == "ambiguous"
                                          else "judge_request_error"),
                    "judge": {"model": judge_cfg["model"],
                              "messages": msgs,
                              "error": f"{type(e).__name__}: {e}"},
                    "units": [], "evidence_summary":
                        "judge request failed; NOT an adherence verdict",
                    "recorded_at": common.utc_now()})
            n_err += 1
            print(f"  [{i}/{len(plan)}] {rid}: request {kind} error "
                  f"({type(e).__name__})")
            continue

        usage = result["usage"]
        cost = budget_mod.actual_cost(judge_cfg, usage)
        ledger.commit(reservation["reservation_id"], cost_usd=cost,
                      usage=usage, request_id=result["request_id"])

        if result["finish_reason"] == "length":
            status, parsed = "judge_output_truncated", None
            verdict = "uncertain"
        else:
            status, parsed = jc.parse_verdict(result["raw_output"])
            verdict = parsed["verdict"] if parsed else "uncertain"

        key = ad.eval_key(rid, rec, "llm_judge",
                          base["rubric_sha256"], psha,
                          judge_cfg["model"], lex_fp)
        common.append_jsonl(evals_path, {
            **base, "eval_key": key, "verdict": verdict,
            "processing_status": status,
            "judge": {
                "model": judge_cfg["model"],
                "model_returned": result["model_returned"],
                "judge_config_sha256": judge_cfg_sha,
                "request_params": client.request_params(),
                "messages": msgs,
                "raw_output": result["raw_output"],
                "parsed_verdict": parsed,
                "usage": usage,
                "cost_usd": cost,
                "request_id": result["request_id"],
                "reservation_id": reservation["reservation_id"],
            },
            "units": det.get("units", []),
            "evidence_summary": (parsed or {}).get("evidence") or "",
            "recorded_at": common.utc_now()})
        n_done += 1
        print(f"  [{i}/{len(plan)}] {rid}: {verdict} "
              f"({status}, ${cost:.4f})")
    print(f"done: {n_done} judged, {n_err} request errors")
    return 0 if n_err == 0 else 2


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_args(p):
        p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
        p.add_argument("--conditions", nargs="+", default=None)
        p.add_argument("--lexicon", type=Path, default=None,
                       help="override lexicon path (tests/preview)")

    r = sub.add_parser("run", help="deterministic evaluation (offline)")
    common_args(r)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="eval coverage summary (offline)")
    common_args(s)
    s.set_defaults(fn=cmd_status)

    jp = sub.add_parser("judge-preview",
                        help="preview judge requests + cost (offline)")
    common_args(jp)
    jp.add_argument("--judge-config", type=Path,
                    default=jc.DEFAULT_JUDGE_CONFIG)
    jp.add_argument("--judge-dir", type=Path, default=jc.JUDGE_DIR)
    jp.add_argument("--max-input-tokens-est", type=int,
                    default=jc.DEFAULT_MAX_INPUT_TOKENS_EST)
    jp.set_defaults(fn=cmd_judge_preview)

    jr = sub.add_parser("judge-run",
                        help="execute judge calls (PAID, ledger-guarded)")
    common_args(jr)
    jr.add_argument("--judge-config", type=Path,
                    default=jc.DEFAULT_JUDGE_CONFIG)
    jr.add_argument("--judge-dir", type=Path, default=jc.JUDGE_DIR)
    jr.add_argument("--purpose", default=PURPOSE_PILOT,
                    choices=[PURPOSE_VALIDATION, PURPOSE_PILOT],
                    help="ledger run_id tag separating validation from "
                         "pilot judging in cost reports")
    jr.add_argument("--budget-config", type=Path,
                    default=DEFAULT_BUDGET_CONFIG)
    jr.add_argument("--ledger", type=Path, default=None)
    jr.add_argument("--max-input-tokens-est", type=int,
                    default=jc.DEFAULT_MAX_INPUT_TOKENS_EST)
    jr.set_defaults(fn=cmd_judge_run)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
