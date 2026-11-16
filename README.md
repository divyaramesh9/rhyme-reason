# Poetic Constraints on LLM Reasoning

A study of how poetic constraints on an LLM's *visible* reasoning affect
answer accuracy. Two benchmarks are prepared locally; responses are
generated under poetic and control conditions, adherence is judged
manually, and accuracy is compared on held-out questions.

| Dataset | Task | Source |
|---|---|---|
| **GSM8K** | grade-school math word problems | [openai/grade-school-math](https://github.com/openai/grade-school-math) |
| **LogiQA v2** | logical multiple-choice reading comprehension | [csitfun/LogiQA2.0](https://github.com/csitfun/LogiQA2.0) |

## Documentation

| file | contents |
|---|---|
| `docs/data_prep.md` | dataset schemas, splits, pilot manifests, validation, licensing, caveats |
| `docs/conditions.md` | device rules, grouping requirements, known overlaps |
| `docs/two_stage_design.md` | full screening → review → freeze → evaluation methodology |
| `docs/blank_line_audit.md` | whitespace-preservation audit of the response path |

## Environments

Requires Python 3.10+.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install -r requirements-generation.txt
```

- `.venv` — stdlib tools + tests only (offline scripts run under plain
  `python3` too, as they are stdlib-only).
- `.venv-mlx` — generation: `mlx-lm` (Apple silicon) + `openai`.

## Data preparation

```bash
python3 scripts/prepare_data.py --download   # once; re-verify + prepare later without the flag
python3 -m pytest tests/ -v                  # tests
```

Standardized `*.inputs.jsonl` / `*.answers.jsonl` per split under
`data/processed/`; pilot manifests under `data/manifests/`. Details:
`docs/data_prep.md`.

## Generating responses

One command for all backends and stages:

```bash
.venv-mlx/bin/python scripts/run_generation.py \
    --manifest data/manifests/screening4.json \
    --conditions answer_only prose_reasoning \
    --model-config configs/model_qwen2.5-7b-instruct-4bit.json \
    --output-dir outputs/<run_name> --stage <stage>
```

Key flags: `--dry-run` (render prompts + counts, no model/network/writes),
`--limit-per-dataset N`, `--stage screening|controls|evaluation`,
`--max-new-tokens 1024` (default cap, identical for all models),
`--budget-config` for paid runs.

- Prompts: `prompts/system.txt` + `prompts/<condition>.txt`, rendered through
  the model's chat template (local) or as API messages (OpenAI).
- Each completed response is appended to `<output-dir>/responses.jsonl`
  immediately; failures go to `failures.jsonl`.
- **Resume**: rerun the same command — completed (example, condition) pairs
  are skipped. Changed model/prompts/manifest/settings → use a new
  `--output-dir`; resuming incompatible output is refused.

## Conditions

Controls: `answer_only`, `prose_reasoning`, `line_breaks`.
Poetic devices (reasoning only): `rhyme_scheme`, `syllable_counts`,
`alliteration`, `anaphora`, `epistrophe`, `meter`, `couplets`,
`stanza_structures`. Rules and overlaps: `docs/conditions.md`.

## Scoring

```bash
python3 scripts/score_answers.py --run-dir outputs/<run_name>
```

Strict `FINAL_ANSWER:` parsing; missing/malformed/ambiguous finals and
generation failures stay in the denominator. Writes `<run-dir>/scores.json`.

## Records, exports, inspection

Each response record keeps `raw_response` verbatim and `reasoning_text` as a
literal slice between `REASONING:`/`FINAL_ANSWER:` (whitespace preserved),
plus `stage`, `messages`, model/provider/token/finish metadata. Full field
list: `docs/two_stage_design.md`.

```bash
python3 scripts/export_responses.py --run-dir <run> --export-root exports   # one .txt per response
python3 scripts/inspect_response.py --run-dir <run> --scan                  # blank-line audit
python3 scripts/inspect_response.py --run-dir <run> --example-id <id> --condition <c>
python3 scripts/extract_reasoning.py --run-dir <run>   # non-destructive backfill of old records
```

## Two-stage study

Full methodology: `docs/two_stage_design.md`.

**Stage 1 — screening.** `data/manifests/screening4.json` = 4 questions
(`gsm8k-train-00204`, `gsm8k-train-00244`, `logiqa_v2-validation-00051`,
`logiqa_v2-validation-00061`) × 8 devices × 3 models = **96 responses**
(+36 controls). A feasibility screen, not a population estimate.

**Stage 2 — evaluation.** Manually label adherence, freeze a retained
device set, then compare accuracy on a content-disjoint test manifest.

| model | config | backend |
|---|---|---|
| Qwen2.5-7B-Instruct-4bit (pinned) | `configs/model_qwen2.5-7b-instruct-4bit.json` | `mlx_local` |
| Llama-3.1-8B-Instruct-4bit (pinned) | `configs/model_llama-3.1-8b-instruct-4bit.json` | `mlx_local` |
| GPT-4o `gpt-4o-2024-08-06` | `configs/model_gpt-4o-2024-08-06.json` | `openai` |

Workflow commands (all offline unless noted):

```bash
python3 scripts/check_local_setup.py --model-config configs/model_llama-3.1-8b-instruct-4bit.json
python3 scripts/check_openai_access.py --model-config configs/model_gpt-4o-2024-08-06.json
python3 scripts/estimate_cost.py --manifest <m> --conditions ... --model-config <gpt4o config>
python3 scripts/prepare_review.py --run-dirs <runs...> --out-dir reviews/<name>
python3 scripts/import_review.py --csv reviews/<name>/review.csv
python3 scripts/plot_review.py --summary reviews/<name>/review.summary.json  # needs matplotlib
python3 scripts/freeze_selection.py freeze        # after editing configs/device_selection.json
python3 scripts/build_eval_manifest.py --size-per-dataset N
python3 scripts/compare_accuracy.py --run-dirs <eval runs...>
```

OpenAI spending is ledger-guarded (`outputs/openai_ledger.jsonl`):
$5 preliminary cap, $20 total cap, conservative reservation before each
request. Inspect with `python3 scripts/budget.py --report`.

## Layout

```
configs/            # model configs, budget.json, device_selection.json
prompts/            # system.txt + one template per condition
scripts/            # all pipeline commands (stdlib unless noted)
data/manifests/     # pilot.json, pilot_disjoint.json, screening4.json, sources.json
data/processed/     # standardized inputs/answers (gitignored)
outputs/            # run dirs + openai_ledger.jsonl (gitignored)
exports/, reviews/  # readable exports, manual review packs (gitignored)
docs/               # methodology and stage detail
tests/              # hardware-independent tests
```
