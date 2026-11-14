"""Tests for visible-response reasoning extraction (response_parsing.py).

Covers whitespace preservation, answer-only handling, missing/repeated
markers, and truncated (unfinished) reasoning traces. Hardware-free.
"""

import response_parsing as rp


# --------------------------------------------------------------------------
# Whitespace preservation — extracted reasoning is a literal slice
# --------------------------------------------------------------------------

def test_reasoning_whitespace_preserved_exactly():
    raw = ("REASONING:\n  indented step  \n\nblank above\n"
           "trailing spaces   \n\n\nFINAL_ANSWER: 42\n")
    f = rp.extract_reasoning_fields(raw, expects_reasoning=True)
    assert f["reasoning_parse_status"] == "extracted"
    assert f["reasoning_text"] == (
        "\n  indented step  \n\nblank above\ntrailing spaces   \n\n\n")
    assert f["final_answer_text"] == "42"
    assert f["final_answer_status"] == "ok"


def test_reasoning_not_stripped_or_normalized():
    raw = "REASONING:\n\n\n   spaced\nFINAL_ANSWER: B"
    f = rp.extract_reasoning_fields(raw, True)
    assert f["reasoning_text"] == "\n\n\n   spaced\n"


def test_final_answer_only_field_payload():
    raw = "REASONING:\nwork\nFINAL_ANSWER:   72.5  "
    f = rp.extract_reasoning_fields(raw, True)
    assert f["final_answer_text"] == "72.5"


# --------------------------------------------------------------------------
# answer_only — reasoning not applicable
# --------------------------------------------------------------------------

def test_answer_only_reasoning_not_applicable():
    f = rp.extract_reasoning_fields("FINAL_ANSWER: 72",
                                    expects_reasoning=False)
    assert f["reasoning_text"] is None
    assert f["reasoning_parse_status"] == "not_applicable"
    assert f["final_answer_text"] == "72"


def test_answer_only_extra_text_kept_raw():
    """Unexpected explanation stays in raw_response for the scorer's
    extra-text flag; reasoning_text still null."""
    raw = "Let me think: 48+24=72.\nFINAL_ANSWER: 72"
    f = rp.extract_reasoning_fields(raw, expects_reasoning=False)
    assert f["reasoning_text"] is None
    assert f["reasoning_parse_status"] == "not_applicable"
    # the raw text itself is untouched upstream; nothing is lost here
    assert rp.record_raw_text({"raw_response": raw}) == raw


# --------------------------------------------------------------------------
# Missing / repeated markers
# --------------------------------------------------------------------------

def test_missing_reasoning_marker():
    f = rp.extract_reasoning_fields("prose without header\n"
                                    "FINAL_ANSWER: 5", True)
    assert f["reasoning_parse_status"] == "missing_reasoning_marker"
    assert f["reasoning_text"] is None
    assert f["final_answer_text"] == "5"


def test_missing_final_answer_keeps_partial_trace():
    raw = "REASONING:\nstep one\nstep two trails off with no answer"
    f = rp.extract_reasoning_fields(raw, True)
    assert f["reasoning_parse_status"] == "partial_missing_final_answer"
    assert f["reasoning_text"] == "\nstep one\nstep two trails off with no answer"
    assert f["final_answer_text"] is None
    assert f["final_answer_status"] == "missing"


def test_truncated_mid_reasoning_preserved():
    """A response cut by the token cap inside REASONING keeps the partial
    trace verbatim rather than being discarded."""
    raw = "REASONING:\nline one rhymes\nline two al"
    f = rp.extract_reasoning_fields(raw, True)
    assert f["reasoning_parse_status"] == "partial_missing_final_answer"
    assert f["reasoning_text"].endswith("line two al")


def test_repeated_reasoning_marker_is_ambiguous():
    raw = "REASONING:\na\nREASONING:\nb\nFINAL_ANSWER: 1"
    f = rp.extract_reasoning_fields(raw, True)
    assert f["reasoning_parse_status"] == "ambiguous_markers"
    assert f["reasoning_text"] is None  # raw_response preserved by record


def test_repeated_identical_final_answer_ok():
    raw = "REASONING:\nwork\nFINAL_ANSWER: 7\nnote\nFINAL_ANSWER: 7"
    f = rp.extract_reasoning_fields(raw, True)
    assert f["reasoning_parse_status"] == "extracted"
    assert f["final_answer_text"] == "7"
    assert f["final_answer_values"] == ["7", "7"]
    # reasoning stops at the first FINAL_ANSWER marker
    assert f["reasoning_text"] == "\nwork\n"


def test_conflicting_final_answers_flagged_not_guessed():
    raw = "REASONING:\nwork\nFINAL_ANSWER: 7\nFINAL_ANSWER: 8"
    f = rp.extract_reasoning_fields(raw, True)
    assert f["final_answer_status"] == "ambiguous"
    assert f["final_answer_text"] is None
    assert f["final_answer_values"] == ["7", "8"]
    # reasoning boundary itself is still deterministic
    assert f["reasoning_text"] == "\nwork\n"


def test_marker_mentions_inside_reasoning_line_dont_split():
    """'FINAL_ANSWER:' mid-line inside reasoning is not a marker."""
    raw = ("REASONING:\nI will write FINAL_ANSWER: later\n"
           "done\nFINAL_ANSWER: 9")
    f = rp.extract_reasoning_fields(raw, True)
    assert f["final_answer_status"] == "ok"
    assert f["final_answer_text"] == "9"


# --------------------------------------------------------------------------
# Schema compatibility helper
# --------------------------------------------------------------------------

def test_record_raw_text_both_schemas():
    assert rp.record_raw_text({"raw_response": "x"}) == "x"
    assert rp.record_raw_text({"response": "y"}) == "y"
    assert rp.record_raw_text({"raw_response": "x",
                               "response": "y"}) == "x"
    assert rp.record_raw_text({}) == ""
