"""Shared constants for the two-stage poetic-reasoning study.

Kept tiny on purpose: condition names live in prompts/<condition>.txt and
are discovered from disk by prompt_lib; this file only names the fixed
experiment vocabulary used by several scripts and tests.
"""

# The eight poetic devices under screening. Each name must correspond to
# prompts/<name>.txt. `line_breaks` is NOT a device -- it is the
# line-structure control (see docs/conditions.md).
POETIC_DEVICES = [
    "rhyme_scheme",
    "syllable_counts",
    "alliteration",
    "anaphora",
    "epistrophe",
    "meter",
    "couplets",
    "stanza_structures",
]

# Control conditions, kept separate from the poetic devices.
CONTROL_ANSWER_ONLY = "answer_only"
CONTROL_PROSE_COT = "prose_reasoning"   # regular prose chain-of-thought
CONTROL_LINE_BREAKS = "line_breaks"     # line structure, no poetic device
CONTROLS = [CONTROL_ANSWER_ONLY, CONTROL_PROSE_COT, CONTROL_LINE_BREAKS]

# Run stages. "development" covers historical/debug runs.
STAGES = ["development", "screening", "controls", "evaluation"]

# Allowed manual adherence labels for the review CSV (empty = unlabeled).
REVIEW_LABELS = ["pass", "fail", "uncertain", "not_evaluable"]

DEFAULT_MAX_NEW_TOKENS = 1024
