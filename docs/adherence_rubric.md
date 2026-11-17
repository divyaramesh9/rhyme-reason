# Automated adherence rubric (v1)

Operational spec for `scripts/adherence.py` + the LLM judge
(`scripts/judge_client.py`, templates in `judge/`). It defines, for each
frozen device, which lines are evaluated, the unit of comparison, and
what produces `pass` / `fail` / `uncertain` / `not_evaluable`. The
per-device text the judge sees lives in `judge/rubrics/<device>.txt`
(sha256'd into every eval record).

Status: **development rubric for the pilot.** Ambiguities that could not
be settled mechanically are listed under *Open questions* rather than
silently resolved.

## Shared policies

**Evaluated text (derived view).** The evaluator looks at a *reasoning
view*, never at a normalized copy of stored data:

- if `reasoning_text` exists, it is evaluated verbatim (whitespace
  preserved);
- if the `REASONING:` marker is missing but the response contains
  visible reasoning (observed in screening: a model emitted bare
  reasoning lines with no header), the view is the raw response up to
  the first `FINAL_ANSWER:` marker — this is what human reviewers
  judged;
- ambiguous markers / `answer_only` → `not_evaluable`.

The record is never modified; the view is derived per evaluation.

**Applicable lines.** Non-blank lines of the view, in order.
Whitespace-only lines are separators, not units. A line that looks like
a heading or label inside reasoning still counts as a reasoning line —
it is visible reasoning content. No vacuous pass: zero applicable lines
→ `not_evaluable`, never `pass`.

**Truncation.** Responses that hit the 1,024-token cap or lack
`FINAL_ANSWER:` are still evaluated on their visible lines; the
`finish_reason`/`hit_output_limit` flags travel in the record. Judge
input is never truncated to save tokens — oversized views are flagged
`judge_skipped_too_long` instead.

**Verdicts.** `pass` = every applicable unit satisfies the rule;
`fail` = at least one unit clearly violates it; `uncertain` = a genuine
ambiguity prevents a decision; `not_evaluable` = nothing to evaluate.
Processing statuses (`judge_output_invalid`, `judge_output_truncated`,
`judge_request_error`, `judge_unresolved`, `judge_skipped_too_long`)
are pipeline events, kept strictly separate from verdicts and never
counted as failures.

## anaphora — deterministic

- **Rule:** every applicable line begins with the supplied phrase
  `We know ` (frozen, includes a trailing space).
- **Case:** case-sensitive; `we know` fails.
- **Boundary:** leading whitespace ignored; the phrase must start on a
  word boundary, so `We known…` fails.
- **Phrase-only lines (user decision):** a line containing ONLY the
  phrase (`We know`, `We know.`, `We know,`) carries no reasoning
  content → **fail**. The reasoning must say something after the
  phrase.
- **Punctuation:** none inside the phrase; trailing content
  unrestricted.
- **Success:** all applicable lines comply.
- **Judge:** never needed (literal check).

## epistrophe — deterministic

- **Rule:** every applicable line ends with the supplied phrase
  `as we see`.
- **Case:** case-sensitive.
- **Boundary:** the phrase must start on a word boundary —
  `…ideas we see` does **not** count (the `as` belongs to `ideas`).
- **Whitespace/punctuation (user decision):** the phrase must be
  the last **words** on the line; trailing whitespace ignored. Any
  reasonable punctuation may follow it — `as we see.` ✓, `as we see,` ✓
  (as in normal verse), `;`/`:`/`!`/`?`/dashes/closing quotes ✓, bare
  `as we see` ✓ — but words, digits, or stray symbols after the phrase
  fail (`as we see now` ✗, `as we see@` ✗). Tolerated punctuation is
  flagged per line.
- **Success:** all applicable lines comply.
- **Judge:** never needed.

## alliteration — deterministic (letter rule) + phonetic evidence

- **Rule (frozen wording):** ≥3 words per line start with the same
  consonant **letter**.
- **Words:** whitespace tokens with ≥1 letter after edge-punctuation
  stripping; function words count; numbers don't start with a
  consonant letter. `y` is treated as a consonant. Case-insensitive.
- **Letter vs sound:** the code verdict enforces the letter rule as
  written. When a pronunciation lexicon is available it also reports
  whether the letter-matched words share an initial consonant
  *phoneme* — a systematic letter/sound conflict triggers judge review
  under the routing policy (`on_letter_sound_conflict`) because
  initial letters alone are not proof of sound repetition.
- **Success:** every line has ≥3 words sharing one initial consonant
  letter.

## rhyme_scheme — deterministic where resolvable, judge otherwise

- **Rule:** consecutive applicable lines paired (1,2), (3,4), …;
  last words must rhyme; an odd trailing
  line is exempt; rhyming words must be part of the reasoning
  sentences.
- **End word:** last letter-bearing token, edge punctuation stripped,
  lowercased.
- **Rhyme test (lexicon):** rhyme tails — phonemes from the last
  stressed vowel onward — must match under *some* listed pronunciation
  pair → `pass` for that pair. **All other outcomes are
  `unknown` → judge** (user decisions): identical end words may be
  genuine or repetition-for-the-rhyme (judge decides); slant rhymes
  may count (judge decides); OOV words and absent lexicon stay
  `unknown`. Code only confirms clear rhymes — it no longer issues a
  deterministic rhyme fail.
- **"Part of the reasoning sentences":** code cannot verify this; the
  judge checks for bolted-on rhyme words.
- **Success:** every complete pair rhymes with different words.

## meter — judge-decided, code as evidence

- **Rule:** each reasoning line carries an iambic pentameter pattern
  (unstressed–stressed ×5; the frozen prompt says "a iambic pentameter
  pattern", interpreted as iambic pentameter).
- **Code:** reports per-line syllable ranges from the lexicon and fails
  a line only when *no* listed pronunciation combination can reach 10
  syllables. Everything else is `unknown` → judge, because lexical
  stress in context is genuinely ambiguous.
- **Judge:** decides from the stress pattern, not syllable count alone;
  a plausible iambic reading passes a line; ambiguity → `uncertain`.
- **Success:** every line can carry an iambic reading.

## Routing policy (`configs/evaluator_policy.json`)

| device | mode |
|---|---|
| anaphora | deterministic only |
| epistrophe | deterministic only |
| alliteration | deterministic; judge review on letter/sound conflict |
| rhyme_scheme | deterministic for resolved pairs; judge for the rest |
| meter | always judge (code evidence attached) |

The policy file's sha256 is part of every eval key — changing routing
is a new evaluation version, never an overwrite.

## Judge configuration

`configs/judge_gpt-5-mini-2025-08-07.json` (OpenAI, pinned snapshot,
JSON-object output, no temperature — the reasoning-family API rejects
non-default sampling params). Chosen over gpt-4o-mini deliberately:
the judge must not share the GPT-4o family with a generating model.
Access is **not verified** — run `check_openai_access.py` before
judging. Judge input contains only the rule, rubric, reasoning text and
mechanical evidence: no model identity, gold answers, correctness, or
human labels. Question context is not sent for any device (no frozen
rule needs question semantics).

## Open questions — resolved by the user (evaluator v1.1.1)

1. **Phrase-only lines** (a line that is just `We know`): **FAIL** —
   no real reasoning chain. Implemented as a deterministic fail.
2. **Epistrophe trailing punctuation**: **phrase must be the last
   words on the line** — any reasonable punctuation after it counts,
   including a comma (as in normal verse); words, digits, or stray
   symbols after the phrase fail. Bare `as we see` at line end also
   passes.
3. **Slant rhymes and repeated end words**: **allowed, judge decides** —
   slant rhymes may count; repeated end words are allowed when the
   repetition is genuine reasoning rather than just for the rhyme.
   Code no longer fails either case deterministically; both are
   `unknown` → judge.
4. **Meter syllable counting** for numbers/abbreviations: confirmed —
   left to the judge with explicit uncertainty.
5. **Alliteration letter-vs-sound**: confirmed — **letter is primary**
   (easier to evaluate); sound remains relevant supporting evidence,
   so phonetic-conflict routing to the judge stays.
6. **Whole-response strictness**: confirmed — **keep strict
   every-line compliance**; the screening "most lines" convention is
   not adopted by the automated evaluator.
7. **One-line rhyme responses**: confirmed — `<2` applicable lines →
   `uncertain` → judge. (Context: the human-failed screening examples
   did not use line breaks at all, which is why they failed.)

## Validation plan

- `validate_evaluator.py` compares verdicts with the 96 screening
  labels — **evaluator-development data**, reported as implementation
  fidelity, not independent validation.
- `audit_pack.py prepare` draws one pilot response per
  model × dataset × device cell (30), blinded (no verdicts/correctness),
  for the real human audit.
- `audit_pack.py decide` records per-device acceptance; `provisional`
  devices keep a provisional flag in the report.
