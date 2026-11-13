"""Hardware-independent tests for the generation/scoring stage.

Covers prompt construction (no answer leakage), FINAL_ANSWER parsing,
question/condition pairing, resume/config-mismatch logic, and disjoint
pilot selection. Nothing here imports mlx or needs a GPU.
"""

import json
from fractions import Fraction
from pathlib import Path

import pytest

import prompt_lib
import run_generation as rg
import score_answers as sa
import select_disjoint_pilot as sdp

REPO_ROOT = Path(__file__).resolve().parent.parent

GSM8K_REC = {
    "example_id": "gsm8k-train-00001",
    "dataset": "gsm8k",
    "split": "train",
    "source_id": None,
    "question": "Janet has 3 apples and buys 9 more. How many apples?",
    "context": None,
    "options": None,
}
LOGIQA_REC = {
    "example_id": "logiqa_v2-validation-00002",
    "dataset": "logiqa_v2",
    "split": "validation",
    "source_id": 999,
    "question": "Which option weakens the argument?",
    "context": "All cats are mammals. Tom is a cat.",
    "options": {"A": "Tom barks.", "B": "Tom purrs.",
                "C": "Tom flies.", "D": "Tom swims."},
}


def templates():
    return prompt_lib.load_templates(REPO_ROOT / "prompts")


# --------------------------------------------------------------------------
# Prompt construction / answer leakage
# --------------------------------------------------------------------------

def test_question_block_identical_across_conditions():
    t = templates()
    m1 = prompt_lib.build_user_message(GSM8K_REC, "answer_only", t)
    m2 = prompt_lib.build_user_message(GSM8K_REC, "prose_reasoning", t)
    block = prompt_lib.question_block(GSM8K_REC)
    assert block in m1 and block in m2
    assert GSM8K_REC["question"] in m1


def test_logiqa_prompt_includes_passage_and_options():
    t = templates()
    m = prompt_lib.build_user_message(LOGIQA_REC, "answer_only", t)
    assert LOGIQA_REC["context"] in m
    assert LOGIQA_REC["question"] in m
    for label in "ABCD":
        assert f"{label}. {LOGIQA_REC['options'][label]}" in m


def test_prompts_contain_no_answer_material():
    """Gold answers, delimiters, and reference reasoning must not leak."""
    t = templates()
    gold = "72"
    reference = "She sold 48/2 = <<48/2=24>>24 clips."
    for cond in ("answer_only", "prose_reasoning"):
        msgs = prompt_lib.build_messages(GSM8K_REC, cond, t)
        blob = json.dumps(msgs)
        assert "####" not in blob
        assert reference not in blob
        assert "gold" not in blob.lower()
        # The synthetic gold does not appear anywhere in either prompt.
        assert gold not in blob


def test_input_records_have_no_answer_fields():
    for rec in (GSM8K_REC, LOGIQA_REC):
        for field in ("gold_answer", "source_answer",
                      "reference_reasoning", "answer"):
            assert field not in rec


def test_question_condition_pairing():
    """Different (example, condition) pairs yield different messages;
    same pair yields the same question block."""
    t = templates()
    other = dict(GSM8K_REC, example_id="gsm8k-train-00002",
                 question="A different question entirely.")
    a = prompt_lib.build_user_message(GSM8K_REC, "answer_only", t)
    b = prompt_lib.build_user_message(GSM8K_REC, "prose_reasoning", t)
    c = prompt_lib.build_user_message(other, "answer_only", t)
    assert a != b and a != c and b != c
    assert "different question" in c and "different question" not in a
    # Same conversation shape: system + single user turn.
    msgs = prompt_lib.build_messages(GSM8K_REC, "answer_only", t)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_unknown_condition_rejected():
    with pytest.raises(KeyError):
        prompt_lib.build_user_message(GSM8K_REC, "rhyme", templates())


# --------------------------------------------------------------------------
# FINAL_ANSWER parsing
# --------------------------------------------------------------------------

def test_parse_ignores_numbers_in_reasoning():
    raw = ("REASONING:\nShe sold 48/2 = 24 in May, then 48+24 = 72 total, "
           "costing $5.40 of thought.\n\nFINAL_ANSWER: 72")
    status, values = sa.extract_final_answer(raw)
    assert status == "ok" and values == ["72"]
    r = sa.score_response("gsm8k", raw, "72", "prose_reasoning")
    assert r["correct"] and r["normalized_prediction"] == "72"


def test_parse_missing_marker():
    r = sa.score_response("gsm8k", "The answer is 72.", "72",
                          "prose_reasoning")
    assert r["parse_status"] == "missing" and not r["correct"]


def test_parse_ambiguous_conflicting_markers():
    raw = "FINAL_ANSWER: 42\nmore text\nFINAL_ANSWER: 72"
    status, values = sa.extract_final_answer(raw)
    assert status == "ambiguous"
    r = sa.score_response("gsm8k", raw, "72", "prose_reasoning")
    assert r["parse_status"] == "ambiguous" and not r["correct"]


def test_parse_repeated_identical_marker_ok():
    raw = "FINAL_ANSWER: 72\n(echo)\nFINAL_ANSWER: 72"
    assert sa.extract_final_answer(raw)[0] == "ok"


def test_gsm8k_normalization_harmless_formats():
    for s in ("72", " 72 ", "72.", "$72", "1,000", "-5", "0.5"):
        assert sa.normalize_gsm8k(s) is not None
    assert sa.normalize_gsm8k("1,000") == Fraction(1000)
    assert sa.normalize_gsm8k("72.0") == sa.normalize_gsm8k("72")
    for s in ("seventy-two", "72 clips", "1/2", "A", "", "72 or 73"):
        assert sa.normalize_gsm8k(s) is None


def test_gsm8k_exact_numeric_comparison():
    r = sa.score_response("gsm8k", "FINAL_ANSWER: 72.00", "72",
                          "answer_only")
    assert r["correct"]
    r = sa.score_response("gsm8k", "FINAL_ANSWER: 72.1", "72",
                          "answer_only")
    assert not r["correct"] and r["parse_status"] == "parsed"
    r = sa.score_response("gsm8k", "FINAL_ANSWER: seventy-two", "72",
                          "answer_only")
    assert r["parse_status"] == "malformed" and not r["correct"]


def test_logiqa_label_parsing():
    for s, ok in (("A", True), ("b", True), ("  C  ", True),
                  ("D.", True), ("AB", False), ("A or B", False),
                  ("option A", False), ("", False), ("1", False)):
        assert (sa.normalize_logiqa(s) is not None) == ok, s
    r = sa.score_response("logiqa_v2", "FINAL_ANSWER: d", "D",
                          "prose_reasoning")
    assert r["correct"]
    r = sa.score_response("logiqa_v2", "FINAL_ANSWER: A or B", "A",
                          "prose_reasoning")
    assert r["parse_status"] == "malformed"


def test_answer_only_extra_text_flag():
    clean = sa.score_response("gsm8k", "FINAL_ANSWER: 72", "72",
                              "answer_only")
    assert clean["extra_text"] is False
    chatty = sa.score_response(
        "gsm8k", "Let me think. 48+24 is 72.\nFINAL_ANSWER: 72", "72",
        "answer_only")
    assert chatty["extra_text"] is True and chatty["correct"]


# --------------------------------------------------------------------------
# Resume / configuration-mismatch rejection
# --------------------------------------------------------------------------

def _setup_files(tmp_path):
    """Create prompt + manifest files once (only if absent)."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    (prompt_dir / "system.txt").write_text("sys")
    (prompt_dir / "answer_only.txt").write_text("$question_block a")
    (prompt_dir / "prose_reasoning.txt").write_text("$question_block b")
    (prompt_dir / "x.txt").write_text("$question_block x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pools": {}}))
    return prompt_dir, manifest


def _compat(tmp_path, **overrides):
    prompt_dir = tmp_path / "prompts"
    manifest = tmp_path / "manifest.json"
    model_cfg = {"repo_id": "r", "revision": "sha"}
    kw = dict(model_cfg=model_cfg, manifest_path=manifest,
              prompt_dir=prompt_dir, conditions=["answer_only"],
              max_new_tokens=1024, chat_template_sha="ct")
    kw.update(overrides)
    return rg.build_compatibility(**kw)


def test_resume_identical_config_ok(tmp_path):
    _setup_files(tmp_path)
    c = _compat(tmp_path)
    assert rg.check_resume_compat(c, dict(c)) == []


def test_resume_rejects_changed_settings(tmp_path):
    _setup_files(tmp_path)
    base = _compat(tmp_path)
    for mod in ({"max_new_tokens": 512},
                {"chat_template_sha": "other"},
                {"model_cfg": {"repo_id": "r", "revision": "newsha"}}):
        changed = _compat(tmp_path, **mod)
        problems = rg.check_resume_compat(base, changed)
        assert problems, mod


def test_resume_rejects_changed_manifest(tmp_path):
    _setup_files(tmp_path)
    base = _compat(tmp_path)
    # Rewrite the manifest file -> different sha -> must reject.
    (tmp_path / "manifest.json").write_text(
        json.dumps({"pools": {"x": []}}))
    changed = _compat(tmp_path)
    assert rg.check_resume_compat(base, changed)


def test_resume_rejects_changed_prompt_template(tmp_path):
    _setup_files(tmp_path)
    base = _compat(tmp_path)
    (tmp_path / "prompts" / "answer_only.txt").write_text("edited")
    changed = _compat(tmp_path)
    problems = rg.check_resume_compat(base, changed)
    assert any("prompt template" in p for p in problems)


def test_resume_condition_subset_ok_superset_rejected(tmp_path):
    _setup_files(tmp_path)
    base = _compat(tmp_path, conditions=["answer_only", "prose_reasoning"])
    subset = _compat(tmp_path, conditions=["answer_only"])
    # Subset shares no new conditions; template shas identical.
    assert rg.check_resume_compat(base, subset) == []
    superset = _compat(tmp_path,
                       conditions=["answer_only", "prose_reasoning", "x"])
    assert rg.check_resume_compat(base, superset)


# --------------------------------------------------------------------------
# Disjoint pilot selection
# --------------------------------------------------------------------------

def _rec(eid, dataset, question, context=None, options=None):
    return {"example_id": eid, "dataset": dataset, "question": question,
            "context": context, "options": options}


def test_disjoint_excludes_test_overlap_and_refills():
    pool = {
        f"g-{i}": _rec(f"g-{i}", "gsm8k", f"q{i}") for i in range(10)
    }
    pool["g-1"]["question"] = "LEAKED"          # original pick, in test
    test_keys = {("gsm8k", "LEAKED")}
    original = ["g-0", "g-1", "g-2"]
    sel = sdp.select_disjoint_ids(pool, test_keys, original, 3, 42)
    assert "g-1" not in sel["example_ids"]
    assert len(sel["example_ids"]) == 3
    assert sel["excluded"][0]["reason"] == "content_occurs_in_test_split"
    assert len(sel["replacements"]) == 1
    # Deterministic: same inputs -> same output.
    again = sdp.select_disjoint_ids(pool, test_keys, original, 3, 42)
    assert again["example_ids"] == sel["example_ids"]


def test_disjoint_in_sample_dedup_first_wins():
    pool = {
        "a": _rec("a", "gsm8k", "same question"),
        "b": _rec("b", "gsm8k", "same question"),
        "c": _rec("c", "gsm8k", "other"),
        "d": _rec("d", "gsm8k", "third"),
    }
    sel = sdp.select_disjoint_ids(pool, set(), ["a", "b", "c"], 3, 42)
    assert "a" in sel["example_ids"] and "b" not in sel["example_ids"]
    assert sel["excluded"][0]["reason"] == "in_sample_duplicate_content"


def test_disjoint_logiqa_matching_uses_ordered_options():
    q = _rec("x", "logiqa_v2", "q", context="ctx",
             options={"A": "1", "B": "2"})
    same = dict(q, example_id="y",
                options={"A": "1", "B": "2"})
    diff = dict(q, example_id="z",
                options={"A": "2", "B": "1"})  # same texts, other order
    assert sdp.content_key(q) == sdp.content_key(same)
    assert sdp.content_key(q) != sdp.content_key(diff)


def test_disjoint_shortfall_reported():
    pool = {"a": _rec("a", "gsm8k", "q")}
    sel = sdp.select_disjoint_ids(pool, set(), ["a"], 5, 42)
    assert sel["shortfall"] == 4


@pytest.mark.skipif(
    not (REPO_ROOT / "data/processed/gsm8k_train.inputs.jsonl").exists(),
    reason="processed data not prepared",
)
def test_disjoint_manifest_real_data():
    """The written manifest must satisfy both rules against real data."""
    manifest = json.loads(
        (REPO_ROOT / "data/manifests/pilot_disjoint.json").read_text())
    for dataset, pool in manifest["pools"].items():
        inputs = sdp.load_jsonl(
            sdp.split_path(dataset, pool["split"]))
        tests = sdp.load_jsonl(sdp.split_path(dataset, "test"))
        test_keys = {sdp.content_key(r) for r in tests}
        by_id = {r["example_id"]: r for r in inputs}
        keys = []
        for eid in pool["example_ids"]:
            k = sdp.content_key(by_id[eid])
            assert k not in test_keys, eid
            keys.append(k)
        assert len(keys) == len(set(keys))
