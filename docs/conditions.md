# Poetic-constraint conditions — development specification (v1)

Status: **versioned development specifications, not frozen final-study
prompts.** Each condition lives in `prompts/<condition>.txt` and is
independently selectable via `--conditions`.

> **Stage-2 update:** the screening set is the **eight poetic devices**
> `rhyme_scheme`, `syllable_counts`, `alliteration`, `anaphora`,
> `epistrophe`, `meter`, `couplets`, `stanza_structures`
> (`scripts/study_defs.py: POETIC_DEVICES`). `line_breaks` is retained
> as the line-structure *control* alongside `answer_only` and
> `prose_reasoning` — it is not counted as a poetic device. `meter`
> (restored for the screening round) is the older terse template: it
> predates the shared "Do not enumerate / concise lines / labels stay
> plain" scaffolding below, so it is not identical in scaffolding to the
> other seven device prompts. All device rules constrain only the
> reasoning lines.

The seven newer device prompts plus `line_breaks` share the same
scaffolding as `prose_reasoning`:

- identical `$question_block` (question, plus passage and `A.`–`D.` options
  for LogiQA) and the shared neutral system message;
- the shared line-organization instruction: *"Think through the problem
  step by step, putting each reasoning step on a separate line."*, plus
  the shared conciseness instruction: *"Keep the reasoning concise, with
  each line contributing to the solution."*;
- the same `REASONING:` … `FINAL_ANSWER:` output contract — the device
  rule constrains **only the reasoning lines**; `REASONING:` and
  `FINAL_ANSWER:` labels stay plain;
- no fixed total number of lines or groups (the model decides how much
  reasoning it needs); generation is still capped at 1,024 output tokens;
- reasoning is generated **directly** in the constrained form — there is
  no prose-then-rewrite step, and no few-shot examples.

## The line-structured conditions (8 poetic devices + `line_breaks` control)

| condition | exact rule (device constraint on each reasoning line) | scope |
|---|---|---|
| `line_breaks` | none beyond the shared one-step-per-line instruction, plus "Do not enumerate the steps" | — |
| `rhyme_scheme` | lines 1&2 rhyme, lines 3&4 rhyme, and so on (AABB pair rhyme); each pair must use different words, not repetitions; rhyming words must be part of the reasoning sentences; an odd final line is exempt | across lines within each consecutive pair |
| `syllable_counts` | exactly 10 syllables per line | within a line |
| `alliteration` | ≥3 words per line starting with the same consonant letter | within a line |
| `anaphora` | every line begins with the fixed phrase "We know " | within lines, consistent across all lines |
| `epistrophe` | every line ends with the fixed phrase "as we see" | within lines, consistent across all lines |
| `meter` | iambic unstressed-stressed syllable pattern on each reasoning line ("a iambic pentameter pattern" — wording preserved verbatim from the saved template) | within a line |
| `couplets` | reasoning steps paired with a follow-up step; complete pairs; two blank lines separate pairs | across lines within a pair |
| `stanza_structures` | lines grouped into complete four-line stanzas; two blank lines separate stanzas | across lines, structural only |

### Within-line vs across-line

- **Within a line:** `syllable_counts`, `alliteration` — the rule
  is checkable per line in isolation.
- **Across lines:** `rhyme_scheme` (line-final words rhyme within each
  consecutive pair), `anaphora`/`epistrophe` (same opening/closing
  phrase repeated), `couplets`/`stanza_structures` (grouping +
  blank-line separators).
- `line_breaks` introduces only the shared line segmentation and serves
  as the structural control.

### Grouping requirements

- `couplets`: complete pairs separated by two blank lines. Note: as
  specified, pairing is structural — the template does **not** require
  the pair to rhyme.
- `stanza_structures`: complete four-line stanzas separated by two
  blank lines; structure only, no rhyme required.
- `rhyme_scheme` pairs consecutive lines for rhyme purposes (lines 1&2,
  3&4, …) but does not require blank-line separators; a single trailing
  odd line is exempt. Earlier drafts required ABAB four-line groups,
  then one shared monorhyme across all lines — both showed poor
  adherence in smoke runs, so the spec moved to per-pair rhymes.

### Intended comparison with `line_breaks`

`line_breaks` isolates the effect of merely forcing reasoning onto
separate lines. Every other condition adds exactly one further
constraint on top of that shared instruction, so `condition −
line_breaks` estimates the marginal effect of each device while
`line_breaks − prose_reasoning` estimates the effect of line
segmentation itself. All conditions share the question format, system
message, output contract, and decoding settings.

### Known overlaps between devices

- `couplets` ≈ `stanza_structures` with 2-line stanzas (both use blank-line
  separated complete groups).
- `rhyme_scheme` (rhyming consecutive pairs) now overlaps `couplets`
  structurally — both operate on two-line groups; `rhyme_scheme` adds
  the rhyme requirement but no blank-line separator, while `couplets`
  adds the separator but no rhyme.
- `epistrophe` (every line ends with the same word/phrase) would
  trivially satisfy the pair-rhyme requirement on sound while violating
  the "different words within each pair" clause — a scorer must treat
  identical end words as non-compliant for `rhyme_scheme`.
- `anaphora`/`epistrophe` mirror each other and use fixed phrases
  ("We know " / "as we see") rather than model-chosen ones; `epistrophe`'s
  repeated end phrase trivially satisfies the pair-rhyme sound rule.
- `meter` overlaps `syllable_counts` (both constrain per-line length/
  rhythm) but adds a stress pattern. It was briefly removed after smoke
  runs because iambic stress is ambiguous to specify and score, then
  restored for the screening round so adherence can be judged manually
  — that ambiguity is itself a documented limitation, not silently
  resolved.
- `alliteration` is independent of the others and could co-occur with any.

## Open decisions for automatic adherence scoring (not yet implemented)

Answer scoring uses the existing `FINAL_ANSWER:` scorer. **No poetic-
compliance scoring exists yet** and none is claimed; these are the known
hard problems:

- **Number pronunciation:** should "12" count as "twelve" (2 syllables)
  or "one two"? Decimals, "$0.50", fractions, and ranges need a
  consistent verbalization rule before syllable checks.
- **Rhyme checking:** requires mapping line-final words to
  pronunciations (e.g., CMUdict). Slant rhymes, proper nouns, numbers,
  and out-of-vocabulary words need an explicit policy (exact phoneme
  match from last stressed vowel? allow assonance? mark unscorable?).
  Under the current spec, each consecutive pair of reasoning lines must
  share a rhyme sound (pairs may differ from each other), the two
  rhyming words must be different words, and an odd trailing line is
  exempt — a scorer must also define what counts as "part of the
  reasoning sentence" vs an appended padding word.
- **Syllable counting:** dictionary lookup vs estimator; same
  out-of-vocabulary issues.
- **alliteration:** "same consonant letter" (as written) vs "same
  consonant sound" (phonetic) — decide which the scorer enforces;
  function words vs content words.
- **anaphora/epistrophe:** phrases are fixed ("We know " / "as we see"),
  so checking is a literal prefix/suffix match — but the policy for
  case, leading/trailing whitespace, and trailing punctuation after the
  phrase still needs deciding.
- **Grouping checks:** define whether blank lines inside reasoning are
  allowed for non-grouping conditions, and how a truncated final group
  (hit the 1,024-token cap) is counted — currently such responses are
  flagged via `hit_output_limit` and preserved, not repaired.
- **What counts as a reasoning line:** empty lines, markdown artifacts,
  or stray bullets inside `REASONING:` need a policy.

Any future adherence scorer must be documented and versioned alongside
these prompts; do not substitute an LLM judge silently.
