"""Parse visible-response structure: REASONING:/FINAL_ANSWER: markers.

Shared by run_generation.py (per-record fields), extract_reasoning.py
(backfill), and score_answers.py (final-answer scoring). Operates only on
the model's *visible* output text -- it never touches hidden/internal
reasoning and never asks the model to rewrite anything.

All extraction preserves whitespace exactly: no stripping or normalizing
of the captured reasoning text. Line breaks, blank lines, indentation,
and punctuation are part of the experiment.
"""

from __future__ import annotations

import re

# Markers must start a line (same anchoring as the scorer). A marker
# mentioned mid-line inside reasoning text does not count.
REASONING_RE = re.compile(r"^[ \t]*REASONING:[ \t]*", re.MULTILINE)
FINAL_ANSWER_LINE_RE = re.compile(r"^[ \t]*FINAL_ANSWER:[ \t]*(.*?)[ \t]*$",
                                  re.MULTILINE)

# reasoning_parse_status values
STATUS_EXTRACTED = "extracted"
STATUS_PARTIAL_NO_FINAL = "partial_missing_final_answer"
STATUS_MISSING_REASONING = "missing_reasoning_marker"
STATUS_AMBIGUOUS = "ambiguous_markers"
STATUS_NOT_APPLICABLE = "not_applicable"


def record_raw_text(record: dict) -> str:
    """Raw response text from a record of either schema generation.

    records use 'raw_response'; the pilot_baseline run predates
    that field and uses 'response'.
    """
    return record.get("raw_response", record.get("response", ""))


def extract_final_answer(raw: str) -> tuple[str, list[str]]:
    """Collect FINAL_ANSWER: payloads. Returns (status, values).

    status: 'ok' (all found values agree) | 'missing' | 'ambiguous'.
    Values lists every captured payload so repeats are preserved.
    """
    values = [m.strip() for m in FINAL_ANSWER_LINE_RE.findall(raw)]
    if not values:
        return "missing", []
    if len(set(values)) > 1:
        return "ambiguous", values
    return "ok", values


def extract_reasoning_fields(raw: str, expects_reasoning: bool) -> dict:
    """Extract the visible reasoning trace and final-answer field.

    Returns a dict with:
      reasoning_text        -- text strictly between the REASONING: marker
                               and the first FINAL_ANSWER: marker, preserved
                               exactly (null when absent/ambiguous/n-a)
      reasoning_parse_status-- one of the STATUS_* constants
      final_answer_text     -- the unambiguous final-answer payload, or null
      final_answer_values   -- every captured FINAL_ANSWER payload
      final_answer_status   -- 'ok' | 'missing' | 'ambiguous'
    """
    fa_status, fa_values = extract_final_answer(raw)
    final_answer_text = fa_values[0] if fa_status == "ok" else None

    if not expects_reasoning:
        return {
            "reasoning_text": None,
            "reasoning_parse_status": STATUS_NOT_APPLICABLE,
            "final_answer_text": final_answer_text,
            "final_answer_values": fa_values,
            "final_answer_status": fa_status,
        }

    reasoning_markers = list(REASONING_RE.finditer(raw))
    final_markers = list(FINAL_ANSWER_LINE_RE.finditer(raw))

    if len(reasoning_markers) > 1:
        status, reasoning_text = STATUS_AMBIGUOUS, None
    elif not reasoning_markers:
        status, reasoning_text = STATUS_MISSING_REASONING, None
    else:
        closing = [m for m in final_markers
                   if m.start() > reasoning_markers[0].end()]
        if not closing:
            # Reasoning header exists but no FINAL_ANSWER: afterwards --
            # keep the remainder as a partial trace, flag missing marker.
            status = STATUS_PARTIAL_NO_FINAL
            reasoning_text = raw[reasoning_markers[0].end():]
        else:
            status = STATUS_EXTRACTED
            reasoning_text = raw[reasoning_markers[0].end():
                                 closing[0].start()]

    return {
        "reasoning_text": reasoning_text,
        "reasoning_parse_status": status,
        "final_answer_text": final_answer_text,
        "final_answer_values": fa_values,
        "final_answer_status": fa_status,
    }
