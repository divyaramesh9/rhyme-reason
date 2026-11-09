# Poetic Constraints on LLM Reasoning — Dataset Preparation

This stage prepares two benchmarks for a study of how poetic constraints on an
LLM's visible reasoning affect answer accuracy:

| Dataset | Task | Source |
|---|---|---|
| **GSM8K** | grade-school math word problems | [openai/grade-school-math](https://github.com/openai/grade-school-math) |
| **LogiQA v2** | logical reasoning, English multiple-choice reading comprehension (MRC) | [csitfun/LogiQA2.0](https://github.com/csitfun/LogiQA2.0) |

Everything runs locally on CPU with the Python standard library. No model
inference, prompt engineering, judging, or training happens in this stage.

## Installation

Requires Python 3.10+.

```bash
pip3 install -r requirements.txt   # only needed for running the tests (pytest)
```

`scripts/prepare_data.py` itself uses only the standard library and needs no
dependencies.

## Reproducing the prepared data

```bash
# Fetch raw files from pinned commits, verify checksums, prepare, validate:
python3 scripts/prepare_data.py --download

# Later runs can omit --download (raw files are already present):
python3 scripts/prepare_data.py

# Pilot size and seed are configurable:
python3 scripts/prepare_data.py --seed 42 --pilot-size 20
```

`--download` fetches each raw file once, records its SHA-256 in
`data/manifests/sources.json`, and refuses to overwrite a preserved raw file
whose checksum disagrees with the manifest. Every subsequent run re-verifies
the raw files against that manifest before preparing.

## Running checks and tests

```bash
python3 scripts/prepare_data.py     # preparation always ends with validation
python3 -m pytest tests/ -v         # unit tests
```

Validation results land in `reports/data_validation.json`. The script exits
non-zero if any check fails or any source record is malformed; malformed
records are listed explicitly under `unresolved_errors` — nothing is silently
dropped or guessed.

## Layout

```
scripts/prepare_data.py        # download + standardize + pilot + validate
data/raw/                      # untouched upstream downloads (gitignored)
data/processed/                # standardized JSONL outputs (gitignored)
data/manifests/sources.json    # pinned URLs, commits, sha256, license (tracked)
data/manifests/pilot.json      # pilot example IDs (tracked)
reports/data_validation.json   # latest validation report (gitignored)
tests/test_prepare_data.py
```

### Processed files

For each split there are two files joined by `example_id`, plus a metadata
file for LogiQA:

```
data/processed/gsm8k_train.inputs.jsonl        / .answers.jsonl
data/processed/gsm8k_test.inputs.jsonl         / .answers.jsonl
data/processed/logiqa_v2_train.inputs.jsonl    / .answers.jsonl / .metadata.jsonl
data/processed/logiqa_v2_validation.inputs…    / .answers.jsonl / .metadata.jsonl
data/processed/logiqa_v2_test.inputs.jsonl     / .answers.jsonl / .metadata.jsonl
```

**`*.inputs.jsonl`** — model-visible fields only. These files never contain
answers or answer-revealing metadata.

| field | meaning |
|---|---|
| `example_id` | unique ID: `{dataset}-{split}-{NNNNN}` where `NNNNN` is the 0-based line number in the raw file (deterministic; no `hash()`) |
| `dataset` | `gsm8k` or `logiqa_v2` |
| `split` | `train`, `validation`, or `test` |
| `source_id` | upstream identifier if one exists (LogiQA `id` — **not unique upstream**, see caveats); `null` for GSM8K |
| `question` | question text, verbatim |
| `context` | LogiQA reading passage (`text` field upstream); `null` for GSM8K |
| `options` | `{"A": …, "B": …, "C": …, "D": …}` for LogiQA, option text verbatim — labels live in the keys so existing letter-like text (e.g. `"A.I …"`, `"B. t. …"`) is never duplicated; `null` for GSM8K |

**`*.answers.jsonl`** — answer information, kept out of model inputs.

| field | meaning |
|---|---|
| `example_id` | join key |
| `gold_answer` | normalized answer: GSM8K = final integer/decimal as a string (see normalization below); LogiQA = letter `A`–`D` |
| `source_answer` | original answer value preserved unchanged (GSM8K: full solution string incl. `<<…>>` calculator annotations and `####` delimiter; LogiQA: original 0-based index) |
| `reference_reasoning` | GSM8K solution text before the `####` delimiter, verbatim; `null` for LogiQA (no reference explanations exist — none are invented) |

**`*.metadata.jsonl`** (LogiQA only) — `example_id`, `source_id`, and the
upstream `type` reasoning-category annotations. Preserved for possible
stratified analysis; never part of model inputs.

### GSM8K answer handling

The source `answer` field is `explanation + "\n#### " + final`. Extraction uses
only the explicit `####` delimiter — never an arbitrary number from the text.

Normalization rule for `gold_answer`: strip whitespace, remove `,` thousands
separators (`"1,080"` → `"1080"`), and require the result to match
`-?\d+(\.\d+)?`. Anything else (missing/multiple delimiters, non-numeric
finals, fractions) is flagged in `unresolved_errors` and preparation fails —
ambiguous answers are never guessed. In this release all 8,792 finals
normalize cleanly to integers.

### LogiQA answer handling

Upstream `answer` is a **0-based index** into `options` (verified against the
upstream README example and `utils_mrc.py`). It maps explicitly `0→A, 1→B,
2→C, 3→D`. Option order and text are preserved exactly. Out-of-range indices
are flagged as errors.

## Splits: development vs. final evaluation

Official splits are preserved exactly; nothing is merged or moved.

| dataset | raw split | standardized `split` | role |
|---|---|---|---|
| gsm8k | `train` | `train` | development / pilot pool |
| gsm8k | `test` | `test` | final evaluation |
| logiqa_v2 | `train` | `train` | development |
| logiqa_v2 | `dev` | `validation` | development / pilot pool |
| logiqa_v2 | `test` | `test` | final evaluation |

LogiQA's `dev` split is renamed to the standardized name `validation` (the
mapping is recorded in `data/manifests/sources.json` and
`reports/data_validation.json`). No unofficial GSM8K validation split is
created — pilot work uses GSM8K `train` and LogiQA `validation` only.

## Pilot selection

`data/manifests/pilot.json` is the single source of truth for the debugging /
prompt-development pilot: 20 GSM8K `train` + 20 LogiQA `validation` examples
at seed 42 (both configurable via `--pilot-size` / `--seed`).

Procedure: for each dataset, `random.Random(seed).sample(sorted(example_ids),
n)` — a fresh `Random` per dataset, so one dataset's selection never depends
on the other's pool. The manifest stores example IDs only; underlying splits
are unchanged. **Every later experimental condition must read its pilot
questions from this manifest** so all conditions see identical items. The
manifest intentionally contains no timestamp, so identical runs produce
identical bytes. A final held-out test subset has deliberately *not* been
selected yet.

## Validation checks (reports/data_validation.json)

- `raw_checksums` — raw files vs. `sources.json` SHA-256s
- `input_schema` / `answer_schema` — required fields, types, and no
  answer-field leakage into inputs
- `unique_example_ids`
- `input_answer_alignment` — one-to-one, same-order join
- `logiqa_option_mapping` — options keyed A–D, index↔letter consistency,
  `reference_reasoning` null
- `gsm8k_answer_extraction` — all golds normalized numeric, delimiter present
  in `source_answer`
- `record_counts` — prepared counts equal parsed source counts
- `duplicate_questions` — exact-duplicate report (GSM8K: question; LogiQA:
  passage+question+options together), within and across splits. Reported,
  never deleted.
- `pilot_integrity` — pilot IDs belong only to non-test pools, correct sizes,
  no duplicates
- `deterministic_rerun` — previously written outputs byte-identical on rerun

## Data sources, revisions, licensing

| dataset | repository | pinned commit | files used | license |
|---|---|---|---|---|
| GSM8K | github.com/openai/grade-school-math | `3101c7d5072418e28b9008a6636bde82a006892c` | `grade_school_math/data/{train,test}.jsonl` | MIT |
| LogiQA v2 (MRC) | github.com/csitfun/LogiQA2.0 | `955e1d3df6c59d9bfb44d9913da1e1a27ec14e18` | `logiqa/DATA/LOGIQA/{train,dev,test}.txt` | CC BY-NC-SA 4.0 |

Exact raw URLs and per-file SHA-256 checksums: `data/manifests/sources.json`.
The Chinese splits, the NLI (`QA2NLI`) variant, and the FOL/AMR annotation
files of LogiQA2.0 are not used in this stage; LogiQA v1 is not used.

Citations:

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and others},
  journal={arXiv:2110.14168}, year={2021}
}
@ARTICLE{10174688,
  author={Liu, Hanmeng and Liu, Jian and Cui, Leyang and Teng, Zhiyang and
          Duan, Nan and Zhou, Ming and Zhang, Yue},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title={LogiQA 2.0—An Improved Dataset for Logical Reasoning in Natural
         Language Understanding},
  year={2023}, volume={31}, pages={2947--2962},
  doi={10.1109/TASLP.2023.3293046}}
```

## Known caveats (from inspection of the source data)

- **LogiQA `id` is not unique** within its splits (e.g. 12,229 unique IDs over
  12,567 train records — colliding IDs belong to *different* questions), so
  `source_id` alone cannot identify records; `example_id` is position-based
  and always unique.
- **LogiQA contains exact-duplicate items** — 1,349 duplicate groups
  (1,359 extra copies), 443 of them spanning different official splits
  (potential train/test leakage). They are enumerated in
  `reports/data_validation.json` and kept unchanged per instructions;
  downstream evaluation may want to account for them.
- **A few option strings contain letter-like prefixes** that are genuine
  content (`"A.I has become mature…"`, `"B. t. toxins…"`, literal `"A."`–`"D."`
  options). They are preserved verbatim; labels exist only as `options` keys.
- The raw `.txt` files have no trailing newline, so `wc -l` undercounts by
  one; record counts come from JSON parsing.
