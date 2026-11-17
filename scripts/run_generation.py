#!/usr/bin/env python3
"""Resumable generation over manifest examples.

Reads a manifest (data/manifests/*.json) and the processed
*.inputs.jsonl files -- model-input records only; gold answers, reference
reasoning, and source annotations are never loaded here.

One question at a time, fresh conversation per (example, condition),
explicit greedy decoding for local models / temperature 0 for OpenAI,
max 1024 new tokens. Each completed response is appended to
<run>/responses.jsonl immediately; failures go to <run>/failures.jsonl
and are retried on the next resume.

Resuming into an existing run directory is refused when the model,
revision, backend, prompts, settings, or manifest contents differ.

Backends (selected by the model config, configs/model_*.json):
  mlx_local -- local MLX-LM (Qwen, Llama); pinned snapshot revision.
  openai    -- OpenAI Chat Completions; sequential requests, no SDK
               retries, persistent budget ledger with caps
               (scripts/budget.py, configs/budget.json).

Usage:
    python3 scripts/run_generation.py \
        --manifest data/manifests/screening4.json \
        --conditions rhyme_scheme couplets \
        --model-config configs/model_llama-3.1-8b-instruct-4bit.json \
        --output-dir outputs/screening_llama \
        --stage screening --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import backends
import budget as budget_mod
import common
import prompt_lib
import response_parsing as rp
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT / "configs" / "model_qwen2.5-7b-instruct-4bit.json"
)
DEFAULT_BUDGET_CONFIG = REPO_ROOT / "configs" / "budget.json"

RESPONSES_FILE = "responses.jsonl"
FAILURES_FILE = "failures.jsonl"
CONFIG_FILE = "run_config.json"
INVOCATIONS_FILE = "invocations.jsonl"

MEMORY_FIELD_DOC = {
    "mlx_peak_memory_bytes": (
        "mx.get_peak_memory() (mx.metal.get_peak_memory() on older mlx) -- "
        "peak MLX Metal-allocator memory since reset_peak_memory() was "
        "called immediately before this generation (per-response peak, "
        "bytes)."
    ),
    "mlx_active_memory_bytes": (
        "mx.get_active_memory() -- MLX memory in use right after this "
        "generation (bytes)."
    ),
    "mlx_cache_memory_bytes": (
        "mx.get_cache_memory() -- MLX Metal cache retained after this "
        "generation (bytes)."
    ),
    "generation_response_peak_gib": (
        "peak_memory field of the final mlx_lm GenerationResponse; MLX "
        "reports it in GiB."
    ),
    "process_maxrss_bytes": (
        "resource.getrusage(RUSAGE_SELF).ru_maxrss -- peak resident set of "
        "the whole process; on macOS this is bytes (monotonic, not "
        "per-response)."
    ),
}

# Keys absent from older run_config.json files get these implied values
# when checking resume compatibility, so pre-backend runs can still
# resume under the new code.
IMPLIED_COMPAT_DEFAULTS = {"backend": "mlx_local"}


# --------------------------------------------------------------------------
# Work items
# --------------------------------------------------------------------------

def load_manifest_examples(manifest_path: Path,
                           limit_per_dataset: int | None,
                           processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    """Expand a pilot manifest into ordered input records (model-visible
    fields only -- answers are never read here).

    --limit-per-dataset takes the first N manifest IDs per dataset, so the
    same example set is used for every requested condition.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = []
    for dataset, pool in manifest["pools"].items():
        split = pool["split"]
        inputs_path = processed_dir / f"{dataset}_{split}.inputs.jsonl"
        by_id = {r["example_id"]: r for r in common.read_jsonl(inputs_path)}
        ids = pool["example_ids"]
        if limit_per_dataset is not None:
            ids = ids[:limit_per_dataset]
        for eid in ids:
            try:
                examples.append(by_id[eid])
            except KeyError:
                raise SystemExit(
                    f"error: manifest id {eid!r} not found in {inputs_path}"
                )
    return examples


# --------------------------------------------------------------------------
# Run configuration / resume
# --------------------------------------------------------------------------

def build_compatibility(model_cfg: dict, manifest_path: Path,
                        prompt_dir: Path, conditions: list[str],
                        max_new_tokens: int, chat_template_sha: str) -> dict:
    """Fields whose change makes resuming into this run invalid."""
    try:
        manifest_display = str(manifest_path.relative_to(REPO_ROOT))
    except ValueError:
        manifest_display = str(manifest_path)
    ident = backends.model_identity(model_cfg)
    return {
        "backend": ident["backend"],
        "model_repo": ident["repo"],
        "model_revision": ident["revision"],
        "manifest_path": manifest_display,
        "manifest_sha256": common.sha256_file(manifest_path),
        "prompt_template_sha256": {
            "system": common.sha256_file(prompt_dir / "system.txt"),
            **{
                c: common.sha256_file(prompt_dir / f"{c}.txt")
                for c in conditions
            },
        },
        "chat_template_sha256": chat_template_sha,
        "max_new_tokens": max_new_tokens,
        "sampler": backends.decoding_spec(model_cfg, max_new_tokens),
        "conditions": sorted(conditions),
    }


def check_resume_compat(stored: dict, current: dict) -> list[str]:
    """Return a list of human-readable mismatches (empty = compatible)."""
    problems = []
    for key, cur in current.items():
        if key == "conditions":
            extra = set(cur) - set(stored.get(key, []))
            if extra:
                problems.append(
                    f"conditions {sorted(extra)} were not part of this run "
                    f"({stored.get(key)}); start a new output directory"
                )
            continue
        if key == "prompt_template_sha256":
            for name, sha in cur.items():
                old = stored.get(key, {}).get(name)
                if old != sha:
                    problems.append(
                        f"prompt template {name!r} changed "
                        f"({old} -> {sha})"
                    )
            continue
        old = stored.get(key, IMPLIED_COMPAT_DEFAULTS.get(key))
        if old != cur:
            problems.append(f"{key}: {old!r} -> {cur!r}")
    return problems


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def dry_run_report(examples: list[dict], conditions: list[str],
                   templates: dict, model_cfg: dict,
                   manifest_path: Path, done: set,
                   stage: str) -> str:
    """Render planned prompts + counts without touching a model."""
    ident = backends.model_identity(model_cfg)
    lines = [
        "DRY RUN -- no model loaded or downloaded; nothing written.",
        f"stage: {stage}",
        f"manifest: {manifest_path} (sha256 {common.sha256_file(manifest_path)[:12]}...)",
        f"backend: {ident['backend']}",
        f"model (not loaded): {ident['repo']}@{ident['revision']}",
        f"system message: {templates['system'].strip()!r}",
        "",
    ]
    work = [
        (rec, cond) for rec in examples for cond in conditions
        if (rec["example_id"], cond) not in done
    ]
    n_pairs = len(examples) * len(conditions)
    lines.append(
        f"planned responses: {n_pairs} "
        f"({len(examples)} examples x {len(conditions)} conditions"
        + (f"; {len(done)} already completed" if done else "")
        + f"; {len(work)} would be generated now)"
    )
    if ident["backend"] == "openai":
        est_in = sum(
            backends.estimate_input_tokens(
                prompt_lib.build_messages(rec, c, templates))
            for rec in examples for c in conditions)
        lines.append(
            "note: OpenAI run -- requests are sequential with a "
            "per-request budget reservation; run estimate_cost.py for a "
            "full offline cost preview."
        )
        lines.append(
            f"approx input tokens over all planned requests: ~{est_in} "
            f"(conservative estimate; see backends.estimate_input_tokens)"
        )
    lines.append("")
    for i, (rec, cond) in enumerate(
            [(r, c) for r in examples for c in conditions], 1):
        skipped = (rec["example_id"], cond) in done
        lines.append(
            f"=== [{i}/{n_pairs}] {rec['example_id']} | {rec['dataset']} "
            f"| {cond}{' [already done]' if skipped else ''} ==="
        )
        lines.append(prompt_lib.build_user_message(rec, cond, templates))
        lines.append("")
    return "\n".join(lines)


def record_failure(path: Path, rec: dict, cond: str, exc: BaseException,
                   extra: dict | None = None) -> None:
    common.append_jsonl(path, {
        "example_id": rec["example_id"],
        "dataset": rec["dataset"],
        "condition": cond,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **(extra or {}),
        "recorded_at": common.utc_now(),
    })


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", type=Path, required=True,
                    help="example manifest (e.g. data/manifests/screening4.json)")
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="condition names = template files in --prompts-dir")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=study_defs.STAGES, default=None,
                    help="run stage recorded in the run config and each "
                         "record (default: manifest's 'stage' field, else "
                         "'development')")
    ap.add_argument("--max-new-tokens", type=int,
                    default=study_defs.DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="cap examples per dataset (first N in manifest order)")
    ap.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG,
                    help="JSON model config; 'backend' selects mlx_local "
                         "or openai")
    ap.add_argument("--prompts-dir", type=Path, default=prompt_lib.PROMPT_DIR)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR,
                    help="directory holding *.inputs.jsonl (default: "
                         "data/processed)")
    ap.add_argument("--budget-config", type=Path,
                    default=DEFAULT_BUDGET_CONFIG,
                    help="cap/ledger config used when backend=openai")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="override ledger path from the budget config")
    ap.add_argument("--dry-run", action="store_true",
                    help="render prompts + planned counts and exit; no "
                         "model load/download, nothing written")
    args = ap.parse_args()

    available = prompt_lib.available_conditions(args.prompts_dir)
    bad = [c for c in args.conditions if c not in available]
    if bad:
        raise SystemExit(
            f"error: unknown condition(s) {bad}; available: {available}"
        )
    templates = prompt_lib.load_templates(args.prompts_dir)
    model_cfg = backends.load_model_config(args.model_config)
    examples = load_manifest_examples(
        args.manifest, args.limit_per_dataset, args.processed_dir)

    manifest_json = json.loads(args.manifest.read_text(encoding="utf-8"))
    stage = args.stage or manifest_json.get("stage") or "development"

    out_dir: Path = args.output_dir
    responses_path = out_dir / RESPONSES_FILE
    failures_path = out_dir / FAILURES_FILE
    config_path = out_dir / CONFIG_FILE

    # Completed (example_id, condition) pairs already on disk -> skip.
    done = {
        (r["example_id"], r["condition"]) for r in common.read_jsonl(responses_path)
    }

    # ---- dry run: render and exit before any model/provider import ----
    if args.dry_run:
        print(dry_run_report(examples, args.conditions, templates,
                             model_cfg, args.manifest, done, stage))
        return 0

    # ---- prepare backend (local snapshot load or provider client) -----
    backend = backends.make_backend(model_cfg)
    t0 = time.perf_counter()
    backend.prepare()
    prepare_seconds = time.perf_counter() - t0

    ident = backends.model_identity(model_cfg)
    ident["revision"] = backend.resolved_revision() or ident["revision"]
    ident["quantization"] = backend.quantization()
    model_meta = ident
    chat_sha = backend.chat_template_sha()

    compat = build_compatibility(model_cfg, args.manifest, args.prompts_dir,
                                 args.conditions, args.max_new_tokens,
                                 chat_sha)
    compat["model_revision"] = ident["revision"]

    # ---- create or validate the run directory ----
    if config_path.exists():
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        problems = check_resume_compat(stored["compatibility"], compat)
        if problems:
            for p in problems:
                print(f"error: resume config mismatch: {p}", file=sys.stderr)
            return 1
        resumed = True
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "run_name": out_dir.name,
            "stage": stage,
            "created_at": common.utc_now(),
            "compatibility": compat,
            "environment": backend.environment_info(MEMORY_FIELD_DOC),
        }
        if getattr(backend, "snapshot_dir", None) is not None:
            config["model_snapshot_dir"] = str(backend.snapshot_dir)
        config_path.write_text(json.dumps(config, indent=2,
                                          ensure_ascii=False) + "\n")
        resumed = False

    work = [
        (rec, cond)
        for rec in examples
        for cond in args.conditions
        if (rec["example_id"], cond) not in done
    ]
    is_openai = ident["backend"] == "openai"
    print(
        f"{'resuming' if resumed else 'new run'}: {out_dir}  (stage: {stage})\n"
        f"  backend: {ident['backend']}\n"
        f"  model: {ident['repo']}@{str(ident['revision'])[:12]}\n"
        f"  prepared in {prepare_seconds:.1f}s\n"
        f"  {len(examples)} examples x {len(args.conditions)} conditions; "
        f"{len(done)} already completed, {len(work)} to generate"
    )
    if not work:
        print("nothing to do")
        return 0

    ledger = None
    if is_openai:
        ledger = budget_mod.BudgetLedger(args.budget_config, args.ledger)
        st = ledger.state()
        print(f"  budget ledger: {ledger.path} "
              f"(committed ${st['committed_usd']:.4f}, open "
              f"${st['open_reserved_usd']:.4f})")

    invocation = {
        "started_at": common.utc_now(),
        "stage": stage,
        "backend": ident["backend"],
        "resumed": resumed,
        "prepare_seconds": round(prepare_seconds, 3),
        "limit_per_dataset": args.limit_per_dataset,
        "conditions": args.conditions,
        "planned": len(work),
        "completed": 0,
        "failed": 0,
    }

    n_done = n_fail = 0
    budget_stop = False
    try:
        for i, (rec, cond) in enumerate(work, 1):
            eid = rec["example_id"]
            reservation = None
            try:
                messages = prompt_lib.build_messages(rec, cond, templates)
                if is_openai:
                    est_in = backends.estimate_input_tokens(messages)
                    est_cost = budget_mod.conservative_cost(
                        model_cfg, est_in, args.max_new_tokens)
                    reservation = ledger.reserve(
                        stage=stage, run_id=out_dir.name,
                        model=model_cfg["model"], example_id=eid,
                        condition=cond, est_cost_usd=est_cost,
                        est_input_tokens=est_in,
                        max_output_tokens=args.max_new_tokens)
                result = backend.generate(messages, args.max_new_tokens)
            except budget_mod.BudgetExceeded as e:
                print(f"\nBUDGET STOP before [{i}/{len(work)}] {eid} "
                      f"{cond}: {e}\ncompleted work is saved; rerun to "
                      f"resume after reviewing the ledger.")
                budget_stop = True
                break
            except Exception as e:
                if reservation is not None:
                    kind = backends.classify_provider_error(e)
                    if kind == "ambiguous":
                        ledger.mark_unresolved(
                            reservation["reservation_id"],
                            run_id=out_dir.name, example_id=eid,
                            condition=cond, error=str(e))
                    else:
                        ledger.void(reservation["reservation_id"],
                                    reason="definitive_provider_error")
                record_failure(failures_path, rec, cond, e, {
                    "budget_reservation_id":
                        reservation["reservation_id"]
                        if reservation else None,
                    "provider_error_class":
                        backends.classify_provider_error(e)
                        if reservation else None,
                })
                n_fail += 1
                print(f"  [{i}/{len(work)}] {eid} {cond}: FAILED "
                      f"({type(e).__name__}: {e})")
                continue

            if reservation is not None and result["provider"]:
                usage = result["provider"]["usage"]
                cost = budget_mod.actual_cost(model_cfg, usage)
                ledger.commit(reservation["reservation_id"],
                              cost_usd=cost, usage=usage,
                              request_id=result["provider"]["request_id"])
                result["provider"]["reservation_id"] = \
                    reservation["reservation_id"]
                result["provider"]["cost_usd"] = cost

            parsed = rp.extract_reasoning_fields(
                result["raw_response"],
                prompt_lib.expects_reasoning(cond, templates))
            record = {
                "example_id": eid,
                "dataset": rec["dataset"],
                "condition": cond,
                "run_id": out_dir.name,
                "stage": stage,
                "model": model_meta,
                "messages": messages,
                **result,
                **parsed,
                "recorded_at": common.utc_now(),
            }
            common.append_jsonl(responses_path, record)
            n_done += 1
            print(
                f"  [{i}/{len(work)}] {eid} {cond}: "
                f"{result['generated_tokens']} tok, "
                f"{result['elapsed_seconds']}s, "
                f"finish={result['finish_reason']}"
            )
    except KeyboardInterrupt:
        print("\ninterrupted; completed responses are saved -- "
              "rerun the same command to resume")

    invocation.update(
        finished_at=common.utc_now(), completed=n_done, failed=n_fail,
        budget_stopped=budget_stop,
    )
    common.append_jsonl(out_dir / INVOCATIONS_FILE, invocation)
    print(f"done: {n_done} completed, {n_fail} failed -> {out_dir}")
    if budget_stop:
        return 3
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
