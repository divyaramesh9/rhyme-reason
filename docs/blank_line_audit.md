# Blank-line audit — response whitespace path (2026-11-14)

**Question:** when `couplets`/`stanza_structures` responses lack the
required blank-line separators, is the model failing to emit them or is
the pipeline deleting them?

**Answer: no destructive operation was found in the response path.** The
pipeline preserves blank lines end to end; the saved records prove it
(single blank lines survive in 122/128 stored responses). The absence of
`\n\n\n` sequences (the "two blank lines" separator those conditions ask
for) is a property of the **generated text itself**, not of storage.

## Path traced

| step | code | whitespace effect |
|---|---|---|
| local decode | `MlxLocalBackend.generate` (`backends.py`): `"".join(r.text for each stream chunk)` | none — concatenation only |
| provider response | `normalize_openai_response`: `message.content` used verbatim; `None` → `""` | none (content never stripped) |
| record build | `run_generation.main`: fields copied into the JSONL record | none |
| JSONL write/read | `append_jsonl` / `read_jsonl` | newlines inside strings become the two-char escape `\n` in the file — encoding, not loss; `line.strip()` only *skips* blank file lines, record text untouched |
| reasoning extraction | `response_parsing.extract_reasoning_fields` | literal substring `raw[m.end(): closing.start()]` — no strip, no normalize |
| final-answer view | `extract_final_answer` / `score_answers.normalize_*` | `.strip()` applied **only to the captured answer payload** — the scorer's normalized view, separate from stored text (intentional, kept) |
| readable export | `export_responses.render_txt` + `write_text(utf-8)` | `"\n".join` embeds raw verbatim; UTF-8 write does not translate newlines on macOS |
| review/export display | `inspect_response.numbered_view`, terminal prints | display-only; stored text untouched |

Other `.strip()` uses audited and cleared: `templates["system"].strip()`
(input-side system message only), `FINAL_ANSWER_RE.sub(...).strip()` in
`score_response` (extra-text flag for `answer_only`, derived flag only),
`line.strip()` guards in every JSONL reader (filtering, not mutating).

## Sample inspection of saved outputs (read-only)

All 5 run directories, 156 records total:

- Empty lines present in `raw_response`: **pilot_baseline 40/80**
  (prose_reasoning paragraph breaks), **poetic_smoke 8/8**,
  **poetic_smoke_all9 35/36**, **poetic_smoke_rhyme_v2 8/8**,
  **poetic_smoke_v3 31/32**.
- `reasoning_text` preserves those blank lines wherever populated
  (e.g. 28/28 in poetic_smoke_v3).
- **Zero** records in any run contain `\n\n\n` — i.e. no saved response
  ever contained two consecutive blank lines, so nothing was available
  to "lose". The model used single blank lines (before `FINAL_ANSWER:`)
  or none inside reasoning.
- No CR/CRLF, no whitespace-only lines, no trailing spaces observed.

Conclusion for the screening round: the separator requirement is a real
model-adherence question, not a parser artifact. Reproduce per-record:

```bash
python3 scripts/inspect_response.py --run-dir outputs/poetic_smoke_v3 --scan
python3 scripts/inspect_response.py --run-dir outputs/poetic_smoke_v3 \
    --example-id gsm8k-train-00204 --condition couplets
```

## Caveat on historical records

The earliest saved text is `raw_response` in `responses.jsonl` — there
is no earlier artifact (streamed token chunks were joined, not logged
individually). Because blank lines demonstrably survive the whole
pipeline, missing separators reflect what the backend surfaced; but for
any *specific* historical response it is logically impossible to
re-prove what the model emitted vs. what any pre-save processing did,
since no earlier copy exists. No fix to past records is attempted; none
is needed — nothing was discarded.

If a destructive bug is ever found later, `scripts/extract_reasoning.py`
already provides the pattern for writing corrected *derived* records
from intact raw text — originals are never overwritten.
