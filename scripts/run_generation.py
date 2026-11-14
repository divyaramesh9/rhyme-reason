#!/usr/bin/env python3
"""Resumable local generation over pilot examples (MLX-LM, Apple silicon).

Reads a pilot manifest (data/manifests/pilot*.json) and the processed
*.inputs.jsonl files -- model-input records only; gold answers, reference
reasoning, and source annotations are never loaded here.

One question at a time, fresh conversation + fresh KV cache per
(example, condition), explicit greedy decoding, max 1024 new tokens.
Each completed response is appended to <run>/responses.jsonl immediately;
failures go to <run>/failures.jsonl and are retried on the next resume.

Resuming into an existing run directory is refused when the model,
revision, prompts, settings, or manifest contents differ.

Usage:
    python3 scripts/run_generation.py \
        --manifest data/manifests/pilot_disjoint.json \
        --conditions answer_only prose_reasoning \
        --output-dir outputs/pilot_baseline \
        --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import prompt_lib
import response_parsing as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT / "configs" / "model_qwen2.5-7b-instruct-4bit.json"
)

RESPONSES_FILE = "responses.jsonl"
FAILURES_FILE = "failures.jsonl"
CONFIG_FILE = "run_config.json"
INVOCATIONS_FILE = "invocations.jsonl"

SAMPLER_SPEC = {
    "name": "greedy",
    "implementation": "mlx_lm.sample_utils.make_sampler(temp=0.0) -> argmax",
    "temperature": 0.0,
    "repetition_penalty": None,
    "top_p": None,
    "top_k": None,
}

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


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
        by_id = {r["example_id"]: r for r in read_jsonl(inputs_path)}
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
    return {
        "model_repo": model_cfg["repo_id"],
        "model_revision": model_cfg["revision"],
        "manifest_path": manifest_display,
        "manifest_sha256": sha256_file(manifest_path),
        "prompt_template_sha256": {
            "system": sha256_file(prompt_dir / "system.txt"),
            **{
                c: sha256_file(prompt_dir / f"{c}.txt")
                for c in conditions
            },
        },
        "chat_template_sha256": chat_template_sha,
        "max_new_tokens": max_new_tokens,
        "sampler": SAMPLER_SPEC,
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
        old = stored.get(key)
        if old != cur:
            problems.append(f"{key}: {old!r} -> {cur!r}")
    return problems


def environment_info(snapshot_dir: Path, chat_template_sha: str) -> dict:
    import importlib.metadata as md

    model_cfg = json.loads((snapshot_dir / "config.json").read_text())
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            p: md.version(p)
            for p in ("mlx", "mlx-lm", "transformers",
                      "huggingface-hub", "tokenizers", "numpy")
            if _has_package(md, p)
        },
        "model_config": {
            "model_type": model_cfg.get("model_type"),
            "quantization": model_cfg.get("quantization"),
            "torch_dtype": model_cfg.get("torch_dtype"),
        },
        "chat_template_sha256": chat_template_sha,
        "memory_field_meanings": MEMORY_FIELD_DOC,
    }


def _has_package(md, name: str) -> bool:
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


def memory_snapshot(gen_response_peak_gib) -> dict:
    """Memory measurements; None where the API is unavailable."""
    snap = {
        "generation_response_peak_gib": gen_response_peak_gib,
        "process_maxrss_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }
    try:
        import mlx.core as mx

        # mlx>=0.32 moved these from mx.metal.* to mx.*; support both
        # without touching the deprecated namespace when unnecessary.
        def get(name):
            if hasattr(mx, name):
                return getattr(mx, name)
            return getattr(mx.metal, name, None)

        peak, active, cache = (get("get_peak_memory"),
                               get("get_active_memory"),
                               get("get_cache_memory"))
        snap["mlx_peak_memory_bytes"] = peak() if peak else None
        snap["mlx_active_memory_bytes"] = active() if active else None
        snap["mlx_cache_memory_bytes"] = cache() if cache else None
    except Exception:
        snap["mlx_peak_memory_bytes"] = None
        snap["mlx_active_memory_bytes"] = None
        snap["mlx_cache_memory_bytes"] = None
    return snap


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def expects_reasoning(condition: str, templates: dict) -> bool:
    """A condition expects a REASONING: section iff its template asks for
    one (the answer_only template does not)."""
    return "REASONING:" in templates["conditions"][condition].template


def dry_run_report(examples: list[dict], conditions: list[str],
                   templates: dict, model_cfg: dict,
                   manifest_path: Path, done: set) -> str:
    """Render planned prompts + counts without touching a model."""
    lines = [
        "DRY RUN -- no model loaded or downloaded; nothing written.",
        f"manifest: {manifest_path} (sha256 {sha256_file(manifest_path)[:12]}...)",
        f"model (not loaded): {model_cfg['repo_id']}@"
        f"{model_cfg['revision']}",
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


def run_one(model, tokenizer, mx, stream_generate, make_sampler,
            messages, max_new_tokens: int) -> dict:
    """Generate one response; returns stats incl. raw text and timing.

    The stream is fully consumed, which forces MLX's lazy evaluation of
    every generated token -- elapsed_seconds therefore covers real
    generation work (prefill + decode), not just kernel enqueueing.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass

    t0 = time.perf_counter()
    parts, prompt_tokens, gen_tokens = [], None, None
    peak_gib, finish_reason, gen_tps, prompt_tps = None, None, None, None
    for r in stream_generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_new_tokens,
        sampler=make_sampler(temp=0.0),
    ):
        parts.append(r.text)
        prompt_tokens = r.prompt_tokens
        gen_tokens = r.generation_tokens
        peak_gib = r.peak_memory
        gen_tps = r.generation_tps
        prompt_tps = r.prompt_tps
        if r.finish_reason:
            finish_reason = r.finish_reason
    elapsed = time.perf_counter() - t0

    return {
        "prompt": prompt_text,
        "raw_response": "".join(parts),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "prompt_tps": prompt_tps,
        "generation_tps": gen_tps,
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": finish_reason,
        "hit_output_limit": finish_reason == "length",
        "memory": memory_snapshot(peak_gib),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest", type=Path, required=True,
                    help="pilot manifest (e.g. data/manifests/pilot_disjoint.json)")
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="condition names = template files in --prompts-dir")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit-per-dataset", type=int, default=None,
                    help="cap examples per dataset (first N in manifest order)")
    ap.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG,
                    help="JSON with repo_id + pinned revision")
    ap.add_argument("--prompts-dir", type=Path, default=prompt_lib.PROMPT_DIR)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR,
                    help="directory holding *.inputs.jsonl (default: "
                         "data/processed)")
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
    model_cfg = json.loads(args.model_config.read_text(encoding="utf-8"))
    examples = load_manifest_examples(
        args.manifest, args.limit_per_dataset, args.processed_dir)

    out_dir: Path = args.output_dir
    responses_path = out_dir / RESPONSES_FILE
    failures_path = out_dir / FAILURES_FILE
    config_path = out_dir / CONFIG_FILE

    # Completed (example_id, condition) pairs already on disk -> skip.
    done = {
        (r["example_id"], r["condition"]) for r in read_jsonl(responses_path)
    }

    # ---- dry run: render and exit before any model/HF import ----------
    if args.dry_run:
        print(dry_run_report(examples, args.conditions, templates,
                             model_cfg, args.manifest, done))
        return 0

    # ---- resolve pinned model snapshot (cached after first download) ----
    from huggingface_hub import snapshot_download
    import mlx.core as mx
    from mlx_lm import load as mlx_load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    t_dl = time.perf_counter()
    snapshot_dir = Path(
        snapshot_download(model_cfg["repo_id"],
                          revision=model_cfg["revision"])
    )
    snapshot_seconds = time.perf_counter() - t_dl

    # ---- load model once ----
    t_load = time.perf_counter()
    model, tokenizer = mlx_load(str(snapshot_dir))
    # Force materialization so the timer reflects real load work on MLX's
    # lazy arrays rather than just parameter declaration.
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - t_load

    chat_template = getattr(tokenizer, "chat_template", None)
    chat_sha = sha256_text(chat_template) if isinstance(
        chat_template, str) else "unavailable"

    model_meta = {
        "repo": model_cfg["repo_id"],
        "revision": model_cfg["revision"],
        "quantization": json.loads(
            (snapshot_dir / "config.json").read_text()
        ).get("quantization"),
    }

    compat = build_compatibility(model_cfg, args.manifest, args.prompts_dir,
                                 args.conditions, args.max_new_tokens,
                                 chat_sha)

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
            "created_at": utc_now(),
            "compatibility": compat,
            "environment": environment_info(snapshot_dir, chat_sha),
            "model_snapshot_dir": str(snapshot_dir),
        }
        config_path.write_text(json.dumps(config, indent=2,
                                          ensure_ascii=False) + "\n")
        resumed = False

    work = [
        (rec, cond)
        for rec in examples
        for cond in args.conditions
        if (rec["example_id"], cond) not in done
    ]
    print(
        f"{'resuming' if resumed else 'new run'}: {out_dir}\n"
        f"  model: {model_cfg['repo_id']}@{model_cfg['revision'][:12]}\n"
        f"  snapshot resolved in {snapshot_seconds:.1f}s; "
        f"model loaded in {load_seconds:.1f}s\n"
        f"  {len(examples)} examples x {len(args.conditions)} conditions; "
        f"{len(done)} already completed, {len(work)} to generate"
    )
    if not work:
        print("nothing to do")
        return 0

    invocation = {
        "started_at": utc_now(),
        "resumed": resumed,
        "snapshot_resolve_seconds": round(snapshot_seconds, 3),
        "model_load_seconds": round(load_seconds, 3),
        "limit_per_dataset": args.limit_per_dataset,
        "conditions": args.conditions,
        "planned": len(work),
        "completed": 0,
        "failed": 0,
    }

    n_done = n_fail = 0
    try:
        for i, (rec, cond) in enumerate(work, 1):
            eid = rec["example_id"]
            try:
                messages = prompt_lib.build_messages(rec, cond, templates)
                result = run_one(model, tokenizer, mx, stream_generate,
                                 make_sampler, messages, args.max_new_tokens)
            except Exception as e:  # failure recorded, run continues
                append_jsonl(failures_path, {
                    "example_id": eid,
                    "dataset": rec["dataset"],
                    "condition": cond,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "recorded_at": utc_now(),
                })
                n_fail += 1
                print(f"  [{i}/{len(work)}] {eid} {cond}: FAILED "
                      f"({type(e).__name__}: {e})")
                continue

            parsed = rp.extract_reasoning_fields(
                result["raw_response"], expects_reasoning(cond, templates))
            record = {
                "example_id": eid,
                "dataset": rec["dataset"],
                "condition": cond,
                "run_id": out_dir.name,
                "model": model_meta,
                **result,
                **parsed,
                "recorded_at": utc_now(),
            }
            append_jsonl(responses_path, record)
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
        finished_at=utc_now(), completed=n_done, failed=n_fail
    )
    append_jsonl(out_dir / INVOCATIONS_FILE, invocation)
    print(f"done: {n_done} completed, {n_fail} failed -> {out_dir}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
