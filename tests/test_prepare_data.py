"""Focused unit tests for scripts/prepare_data.py."""

import json

import pytest

import prepare_data as pd


# --------------------------------------------------------------------------
# GSM8K final-answer extraction
# --------------------------------------------------------------------------

def test_extract_simple_final():
    reasoning, gold = pd.extract_gsm8k_final("Step one.\nStep two.\n#### 72")
    assert gold == "72"
    assert reasoning == "Step one.\nStep two."


def test_extract_removes_thousands_separators():
    _, gold = pd.extract_gsm8k_final("Work.\n#### 1,080")
    assert gold == "1,080".replace(",", "")
    _, gold = pd.extract_gsm8k_final("Work.\n#### 109,200,000")
    assert gold == "109200000"


def test_extract_negative_and_decimal():
    assert pd.extract_gsm8k_final("x\n#### -5")[1] == "-5"
    assert pd.extract_gsm8k_final("x\n#### 0.5")[1] == "0.5"


def test_extract_rejects_missing_delimiter():
    with pytest.raises(pd.MalformedRecordError):
        pd.extract_gsm8k_final("No delimiter here, answer is 42")


def test_extract_rejects_multiple_delimiters():
    with pytest.raises(pd.MalformedRecordError):
        pd.extract_gsm8k_final("#### 3\nmore\n#### 4")


def test_extract_rejects_non_numeric_final():
    with pytest.raises(pd.MalformedRecordError):
        pd.extract_gsm8k_final("work\n#### forty-two")
    with pytest.raises(pd.MalformedRecordError):
        pd.extract_gsm8k_final("work\n#### 1/2")


def test_gsm8k_transform_preserves_source_answer():
    raw = "Natalia sold 48/2 = <<48/2=24>>24 clips.\nTotal <<48+24=72>>72.\n#### 72"
    inputs, answers, errors = pd.transform_gsm8k(
        [{"question": "Q?", "answer": raw}], "train"
    )
    assert not errors
    assert answers[0]["source_answer"] == raw  # unchanged
    assert answers[0]["gold_answer"] == "72"
    assert "####" not in answers[0]["reference_reasoning"]
    assert "<<48/2=24>>" in answers[0]["reference_reasoning"]


def test_gsm8k_malformed_is_flagged_not_dropped():
    inputs, answers, errors = pd.transform_gsm8k(
        [{"question": "Q?", "answer": "no delimiter"}], "test"
    )
    assert inputs == [] and answers == []
    assert len(errors) == 1 and errors[0]["position"] == 0


# --------------------------------------------------------------------------
# LogiQA answer-index -> letter mapping
# --------------------------------------------------------------------------

def test_letter_from_index():
    assert pd.letter_from_index(0, 4) == "A"
    assert pd.letter_from_index(1, 4) == "B"
    assert pd.letter_from_index(2, 4) == "C"
    assert pd.letter_from_index(3, 4) == "D"


def test_letter_from_index_out_of_range():
    with pytest.raises(pd.MalformedRecordError):
        pd.letter_from_index(4, 4)
    with pytest.raises(pd.MalformedRecordError):
        pd.letter_from_index(-1, 4)
    with pytest.raises(pd.MalformedRecordError):
        pd.letter_from_index("0", 4)


def test_logiqa_transform_maps_index_and_preserves_options():
    rec = {
        "id": 218,
        "answer": 3,
        "text": "A passage.",
        "question": "Which option?",
        "options": ["A.", "B.", "D.", "C."],  # verbatim letter-like text kept
        "type": {"Categorical Reasoning": True},
    }
    inputs, answers, metadata, errors = pd.transform_logiqa([rec], "validation")
    assert not errors
    assert answers[0]["gold_answer"] == "D"
    assert answers[0]["source_answer"] == 3
    assert answers[0]["reference_reasoning"] is None
    assert inputs[0]["options"] == {"A": "A.", "B": "B.", "C": "D.", "D": "C."}
    assert inputs[0]["context"] == "A passage."
    assert inputs[0]["source_id"] == 218
    assert metadata[0]["type"] == {"Categorical Reasoning": True}


def test_logiqa_bad_answer_index_flagged():
    rec = {"id": 1, "answer": 9, "text": "t", "question": "q",
           "options": ["a", "b", "c", "d"]}
    inputs, answers, metadata, errors = pd.transform_logiqa([rec], "train")
    assert len(errors) == 1


# --------------------------------------------------------------------------
# Answer fields never reach model inputs
# --------------------------------------------------------------------------

def test_input_records_carry_no_answer_fields():
    g_in, _, _ = pd.transform_gsm8k(
        [{"question": "Q", "answer": "w\n#### 5"}], "train")
    l_in, _, _, _ = pd.transform_logiqa(
        [{"id": 1, "answer": 0, "text": "t", "question": "q",
          "options": ["a", "b", "c", "d"], "type": {}}], "train")
    for rec in g_in + l_in:
        for field in pd.ANSWER_ONLY_FIELDS:
            assert field not in rec
        blob = json.dumps(rec)
        assert "####" not in blob


# --------------------------------------------------------------------------
# Pilot selection determinism
# --------------------------------------------------------------------------

def test_pilot_selection_reproducible():
    pool = [f"gsm8k-train-{i:05d}" for i in range(7473)]
    a = pd.select_pilot_ids(pool, 20, 42)
    b = pd.select_pilot_ids(pool, 20, 42)
    assert a == b
    assert len(a) == 20 and len(set(a)) == 20


def test_pilot_selection_seed_changes_selection():
    pool = [f"gsm8k-train-{i:05d}" for i in range(7473)]
    assert pd.select_pilot_ids(pool, 20, 42) != pd.select_pilot_ids(pool, 20, 7)


def test_pilot_pools_are_independent():
    """Each dataset is drawn with its own Random(seed): changes to one pool
    must not perturb the other dataset's selection."""
    g = [f"gsm8k-train-{i:05d}" for i in range(7473)]
    l = [f"logiqa_v2-validation-{i:05d}" for i in range(1569)]
    base = pd.build_pilot({"gsm8k": g, "logiqa_v2": l}, 20, 42)
    # Perturb the gsm8k pool (drop an element); logiqa picks must be identical.
    perturbed = pd.build_pilot({"gsm8k": g[1:], "logiqa_v2": l}, 20, 42)
    assert perturbed["pools"]["logiqa_v2"]["example_ids"] == \
        base["pools"]["logiqa_v2"]["example_ids"]


def test_pilot_only_from_allowed_splits():
    pools = {
        "gsm8k": [f"gsm8k-train-{i:05d}" for i in range(100)],
        "logiqa_v2": [f"logiqa_v2-validation-{i:05d}" for i in range(100)],
    }
    pilot = pd.build_pilot(pools, 20, 42)
    check = pd.check_pilot(pilot)
    assert check["status"] == "pass"
    for eid in pilot["pools"]["gsm8k"]["example_ids"]:
        assert eid.startswith("gsm8k-train-")
    for eid in pilot["pools"]["logiqa_v2"]["example_ids"]:
        assert eid.startswith("logiqa_v2-validation-")


def test_example_id_deterministic_format():
    assert pd.make_example_id("gsm8k", "train", 0) == "gsm8k-train-00000"
    assert pd.make_example_id("logiqa_v2", "validation", 1568) == \
        "logiqa_v2-validation-01568"


# --------------------------------------------------------------------------
# Duplicate-question reporting (kept, not deleted)
# --------------------------------------------------------------------------

def test_duplicate_questions_reported():
    rec = {"question": "Same?", "answer": "w\n#### 1"}
    inputs, _, _ = pd.transform_gsm8k([rec, dict(rec)], "train")
    result = pd.check_duplicate_questions(inputs)
    assert result["duplicate_groups"] == 1
    assert len(result["groups"][0]["example_ids"]) == 2
    assert len(inputs) == 2  # nothing deleted
