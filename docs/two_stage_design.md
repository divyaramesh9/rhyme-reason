# Two-stage study design (methodology notes)

Stage 1 screens which poetic constraints three models can visibly
follow; stage 2 freezes a retained device set and compares answer
accuracy against controls on held-out questions. All generation is
zero-shot: no demonstrations, no rewriting, no output repair, no LLM
judges. Every response is kept — correct, incorrect, noncompliant,
or truncated.

## Stage 1 — poetic screening (96 responses + optional 36 controls)

- Manifest: `data/manifests/screening4.json` — **4 questions total**
  (2 GSM8K train + 2 LogiQA validation): `gsm8k-train-00204`,
  `gsm8k-train-00244`, `logiqa_v2-validation-00051`,
  `logiqa_v2-validation-00061`. Same 4 for every model and condition.
- Poetic cells: 4 × 8 devices × 3 models = **96**. Controls
  (`answer_only`, `prose_reasoning`, `line_breaks`) on the same
  questions add 36 → 132 total per a full preliminary matrix.
- This is a **feasibility screen, not a population estimate** — four
  examples cannot support adherence-rate claims.
- Screening and control generation are separately selectable via
  `--conditions`; preview counts with `--dry-run` and OpenAI costs with
  `estimate_cost.py` for each set independently.
- Historical outputs may be reused only when questions, prompt hashes,
  model config, and decoding match exactly — check with
  `scripts/check_reuse.py`. Otherwise they remain development results.
  (The existing `poetic_smoke_v3` run covers these same 4 questions × 7
  devices on Qwen — verify before reusing.)

## Models

| model | config | backend | decoding |
|---|---|---|---|
| Qwen2.5-7B-Instruct-4bit | `configs/model_qwen2.5-7b-instruct-4bit.json` | `mlx_local` | greedy (argmax) |
| Llama-3.1-8B-Instruct-4bit | `configs/model_llama-3.1-8b-instruct-4bit.json` | `mlx_local` | greedy (argmax) |
| GPT-4o (snapshot `gpt-4o-2024-08-06`) | `configs/model_gpt-4o-2024-08-06.json` | `openai` | `temperature=0` |

- **GPT-4o status (verified 2026-11-14):** *deprecated* per OpenAI's
  deprecations page — announced 2026-10-14, **shutdown 2027-01-14**;
  still serves API requests until then. A dated snapshot
  (`gpt-4o-2024-08-06`, the API default) is used, not the moving
  `gpt-4o` alias. Suggested replacement `gpt-6-luna` reasons by
  default and is **not** a drop-in substitute. Account access is not
  assumed — check with `scripts/check_openai_access.py` (a metadata
  lookup that generates nothing; it cannot verify billing or quota).
  `temperature=0` is requested but does **not** guarantee identical
  repeated outputs. Max output = 16,384; we cap at 1,024.
- **Llama:** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, pinned
  revision `241a666dad6cb93c8ff213d39a7f34a36bf26db4` (upstream last
  modified 2024-11-26), tokenizer + Llama-3 chat template included in
  the repo. The repo is not gated on HF, but the Llama 3.1 Community
  License still applies. ~4.9 GB weights — fits the 24 GB unified
  memory here (verify on yours with `scripts/check_local_setup.py`).
  Weights are **not** downloaded automatically; download manually.
- Tokenization differs across models; token counts are not
  cross-model-comparable. The 1,024-output-token cap is identical for
  all three.
- Semantic prompt content is identical across models; only
  provider-required serialization differs (local chat template vs API
  messages).

## Manual adherence review

- `scripts/prepare_review.py` writes per-response readable review files
  (question + verbatim device instructions + unaltered raw response +
  a separate diagnostic line-numbered view) and a blank CSV
  (`adherence_label`/`notes` empty): `pass | fail | uncertain |
  not_evaluable`. No gold answers, no correctness in review materials.
- `scripts/import_review.py` validates IDs/labels and summarizes
  model × dataset × device, reporting unanswered and
  uncertain/not_evaluable explicitly.
- **No automatic poetic grader is implemented and no LLM judge is
  used.** Adherence is never inferred from answer correctness.

## Device selection and freezing

- Record decisions in `configs/device_selection.json` (retain/exclude +
  reason per device, source runs/review files, per-model differences,
  unresolved limitations). Selection is based on observed adherence —
  not on which devices score higher on answers — and there is no
  automatic threshold.
- `scripts/freeze_selection.py freeze` validates and writes
  `configs/device_selection.frozen.json` with prompt + selection-file
  hashes; `verify` re-checks drift before final generation. Zero
  retained devices is reported loudly and requires `--allow-empty`.
- One common retained set is used for **all** models; per-model
  differences are documented descriptively only.

## Stage 2 — held-out evaluation

- `scripts/build_eval_manifest.py --size-per-dataset N` builds a test-
  split manifest that is **content-disjoint** from the dev pools
  (GSM8K question text; LogiQA passage + question + ordered options) —
  not just ID-disjoint, because LogiQA duplicates span official splits.
  Sample size is deliberately not defaulted: choose it, then preview
  cost with `estimate_cost.py` before generating.
- Final conditions: `answer_only`, `prose_reasoning`, `line_breaks`,
  plus your frozen devices — identical questions across all models.
- `scripts/score_answers.py` per run, then `scripts/compare_accuracy.py`
  for accuracy by model/dataset and deltas vs each control. Primary
  accuracy uses **all** responses, including noncompliant ones. Final
  adherence/joint success is reported only where manual labels for
  *those* outputs exist — preliminary labels never transfer.
- Keep screening and evaluation in separate manifests and output dirs;
  incomplete runs are reported, never presented as complete.

## OpenAI budget safeguards (`configs/budget.json`)

- Caps: **$5 preliminary bucket** (screening + controls stages
  combined), **$20 total** tracked; ~$5 of the ~$25 credit stays
  outside planned caps as buffer. Caps are limits, never raised
  automatically.
- `outputs/openai_ledger.jsonl` (append-only, persists across runs):
  conservative reservation before each request (estimated input +
  full 1,024-token output at full price — no caching/batch discounts),
  commit of actual usage/cost after each response. Requests are
  sequential, SDK retries disabled (`max_retries=0`), and ambiguous
  failures (timeout/connection/5xx) are recorded as **unresolved** —
  they stay reserved until you review and release them
  (`scripts/budget.py --report`, `--release <id>`).
- A request whose reservation would exceed either cap stops the run
  before sending (exit code 3); nothing is silently resized.
- The ledger tracks **this pipeline only** — not the account balance or
  spending elsewhere. Input-token estimates are approximate
  (conservative chars/3 heuristic + per-message overhead; no o200k
  tokenizer assumed offline).

## Records and resume

- Each response record keeps: `example_id`, `dataset`, `condition`,
  `run_id`, `stage`, `messages` (exact input), `prompt` (rendered local
  prompt where applicable), `model` (repo/revision/quantization/
  backend), unchanged `raw_response`, whitespace-preserving
  `reasoning_text`, `final_answer_text`/`final_answer_values`,
  `reasoning_parse_status`, token counts, `finish_reason`,
  `hit_output_limit`, `provider` (request id, usage, cost for OpenAI),
  `recorded_at`. For `answer_only`, reasoning is null; unexpected
  explanation survives in `raw_response`. Truncated and marker-missing
  outputs are kept and flagged — never retried, extended, or repaired.
- Resume protections are unchanged: completed (example, condition)
  pairs are skipped, never overwritten; changed model/prompts/manifest/
  settings requires a new output directory.
- Exports (`export_responses.py`) and review files embed model /
  dataset / condition / example ID in filenames — collision-free
  across runs and models.
