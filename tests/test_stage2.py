"""Hardware-independent tests for the two-stage study additions.

Covers: end-to-end whitespace preservation (extraction -> JSONL -> export),
the screening4 manifest, budget ledger accounting/caps, mocked OpenAI
provider normalization, dry-run isolation, review CSV round-trip and
import validation, selection freezing, eval-manifest disjointness, and
export filename collisions. All fixtures are synthetic; no model,
network, or credentials are used.
"""

import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import backends
import budget as budget_mod
import build_eval_manifest as bem
import export_responses as ex
import freeze_selection as fs
import import_review as ir
import inspect_response as insp
import prepare_review as pr
import prompt_lib
import response_parsing as rp
import run_generation as rg
import select_disjoint_pilot as sdp
import study_defs

REPO_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================================
# Synthetic fixtures: raw responses exercising every whitespace hazard
# ==========================================================================

# Two-line couplets separated by ONE blank line (text\n\n text).
COUPLET_RAW = (
    "REASONING:\n"
    "The stack is 12 inches thick.\n"
    "Each inch holds 80 pages, so 960 pages total.\n"
    "\n"
    "Six books share those pages evenly.\n"
    "960 divided by 6 gives 160 pages each.\n"
    "\n"
    "FINAL_ANSWER: 160"
)

# Four-line stanzas separated by TWO blank lines (text\n\n\n text).
STANZA_RAW = (
    "REASONING:\n"
    "line one\r\n"
    "line two\r\n"
    "line three\r\n"
    "line four\r\n"
    "\r\n"
    "\r\n"
    "line five\r\n"
    "line six\r\n"
    "line seven\r\n"
    "line eight\r\n"
    "\r\n"
    "FINAL_ANSWER: 42"
)

# Whitespace-only lines, indentation, trailing spaces.
WS_RAW = (
    "REASONING:\n"
    "   \n"
    "  indented step with trailing space   \n"
    "\t\n"
    "plain step\n"
    "\n\n"
    "FINAL_ANSWER: 7"
)

# Blank line immediately after REASONING: and before FINAL_ANSWER:.
ADJACENT_RAW = (
    "REASONING:\n"
    "\n"
    "step\n"
    "\n"
    "FINAL_ANSWER: B\n"
)


def test_couplet_blank_line_preserved_end_to_end(tmp_path):
    """Extraction -> JSONL write/read -> export render: byte identical."""
    f = rp.extract_reasoning_fields(COUPLET_RAW, True)
    assert "\n\n" in f["reasoning_text"]  # the blank separator survives
    assert f["reasoning_parse_status"] == "extracted"

    rec = {"example_id": "e1", "dataset": "gsm8k", "condition": "couplets",
           "run_id": "r", "raw_response": COUPLET_RAW, **f}
    p = tmp_path / "responses.jsonl"
    rg.append_jsonl(p, rec)
    back = rg.read_jsonl(p)[0]
    assert back["raw_response"] == COUPLET_RAW
    assert back["reasoning_text"] == f["reasoning_text"]

    txt = ex.render_txt(back, "Question:\nq", None, {}, "r")
    assert COUPLET_RAW in txt


def test_stanza_double_blank_and_crlf_preserved(tmp_path):
    f = rp.extract_reasoning_fields(STANZA_RAW, True)
    assert "\r\n\r\n\r\n" in f["reasoning_text"]  # two CRLF blank lines
    rec = {"example_id": "e2", "dataset": "gsm8k",
           "condition": "stanza_structures", "raw_response": STANZA_RAW,
           **f}
    p = tmp_path / "r.jsonl"
    rg.append_jsonl(p, rec)
    back = rg.read_jsonl(p)[0]
    assert back["raw_response"] == STANZA_RAW


def test_whitespace_only_indent_trailing_preserved():
    f = rp.extract_reasoning_fields(WS_RAW, True)
    rt = f["reasoning_text"]
    assert "\n   \n" in rt and "\n\t\n" in rt
    assert "trailing space   \n" in rt


def test_blank_lines_adjacent_to_markers_kept():
    f = rp.extract_reasoning_fields(ADJACENT_RAW, True)
    assert f["reasoning_text"] == "\n\nstep\n\n"
    assert f["final_answer_text"] == "B"


def test_reasoning_is_literal_substring():
    for raw in (COUPLET_RAW, STANZA_RAW, WS_RAW, ADJACENT_RAW):
        f = rp.extract_reasoning_fields(raw, True)
        assert f["reasoning_text"] in raw
        assert insp.reasoning_is_exact_slice(
            raw, f["reasoning_text"], True).endswith("YES")


def test_export_raw_section_roundtrip(tmp_path):
    """export_preserves_raw must confirm byte-identical export."""
    run_dir = tmp_path / "outputs" / "runx"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_name": "runx",
        "compatibility": {"model_repo": "m", "model_revision": "abc123",
                          "manifest_path": "x.json"},
    }))
    rec = {"example_id": "gsm8k-train-00001", "dataset": "gsm8k",
           "condition": "couplets", "raw_response": COUPLET_RAW,
           "model": {"repo": "m", "revision": "abc123"}}
    (run_dir / "responses.jsonl").write_text(
        json.dumps(rec) + "\n")
    export_root = tmp_path / "exports"
    dest = (export_root / "runx" / "gsm8k" / "couplets"
            / f"{ex.model_tag(rec['model'], {})}__gsm8k__couplets__"
              "gsm8k-train-00001.txt")
    dest.parent.mkdir(parents=True)
    dest.write_text(ex.render_txt(rec, "q", rec["model"], {}, "runx"),
                    encoding="utf-8")
    msg = insp.export_preserves_raw(run_dir, rec, COUPLET_RAW, export_root)
    assert "byte-identical" in msg


def test_numbered_view_marks_empty_and_wsonly():
    view = insp.numbered_view("a\n\n   \nb  \n")
    lines = view.split("\n")
    assert "<EMPTY>" in lines[1]
    assert "<WS-ONLY" in lines[2]
    assert "<TRAILING-WS" in lines[3]


# ==========================================================================
# Screening manifest
# ==========================================================================

def test_screening4_is_first_two_of_pilot_disjoint():
    s4 = json.loads((REPO_ROOT / "data/manifests/screening4.json")
                    .read_text())
    pd = json.loads((REPO_ROOT / "data/manifests/pilot_disjoint.json")
                    .read_text())
    assert s4["stage"] == "screening"
    total = 0
    for ds, pool in s4["pools"].items():
        assert pool["example_ids"] == pd["pools"][ds]["example_ids"][:2]
        assert pool["split"] == pd["pools"][ds]["split"]
        total += len(pool["example_ids"])
    assert total == 4  # 4 questions total, not per dataset


def test_poetic_devices_and_controls_prompt_files_exist():
    avail = prompt_lib.available_conditions()
    for c in study_defs.POETIC_DEVICES + study_defs.CONTROLS:
        assert c in avail, c
    assert len(study_defs.POETIC_DEVICES) == 8
    assert study_defs.CONTROL_LINE_BREAKS not in study_defs.POETIC_DEVICES


def test_screening_matrix_counts():
    # 4 questions x 8 devices x 3 models = 96 poetic; +36 control = 132.
    s4 = json.loads((REPO_ROOT / "data/manifests/screening4.json")
                    .read_text())
    n_q = sum(len(p["example_ids"]) for p in s4["pools"].values())
    assert n_q * len(study_defs.POETIC_DEVICES) * 3 == 96
    assert n_q * len(study_defs.CONTROLS) * 3 == 36


@pytest.mark.skipif(
    not (REPO_ROOT / "data/processed/gsm8k_train.inputs.jsonl").exists(),
    reason="processed data not prepared")
def test_screening4_ids_exist_and_are_test_disjoint():
    s4 = json.loads((REPO_ROOT / "data/manifests/screening4.json")
                    .read_text())
    for ds, pool in s4["pools"].items():
        inputs = {r["example_id"]: r for r in sdp.load_jsonl(
            sdp.split_path(ds, pool["split"]))}
        test_keys = {sdp.content_key(r) for r in sdp.load_jsonl(
            sdp.split_path(ds, "test"))}
        for eid in pool["example_ids"]:
            assert sdp.content_key(inputs[eid]) not in test_keys


# ==========================================================================
# Budget ledger
# ==========================================================================

def _budget_cfg(tmp_path):
    cfg = tmp_path / "budget.json"
    cfg.write_text(json.dumps({
        "ledger_path": str(tmp_path / "ledger.jsonl"),
        "stage_to_bucket": {"screening": "preliminary",
                            "controls": "preliminary",
                            "evaluation": "evaluation"},
        "bucket_caps_usd": {"preliminary": 5.0},
        "total_cap_usd": 20.0,
    }))
    return cfg


def test_ledger_reserve_commit_totals(tmp_path):
    led = budget_mod.BudgetLedger(_budget_cfg(tmp_path))
    r = led.reserve(stage="screening", run_id="run", model="m",
                    example_id="e", condition="c", est_cost_usd=1.25,
                    est_input_tokens=100, max_output_tokens=1024)
    st = led.state()
    assert st["open_reserved_usd"] == pytest.approx(1.25)
    led.commit(r["reservation_id"], cost_usd=0.01,
               usage={"prompt_tokens": 80, "completion_tokens": 40},
               request_id="req1")
    st = led.state()
    assert st["committed_usd"] == pytest.approx(0.01)
    assert st["open_reserved_usd"] == 0.0
    assert not st["open_reservations"]


def test_ledger_bucket_cap_blocks(tmp_path):
    led = budget_mod.BudgetLedger(_budget_cfg(tmp_path))
    led.reserve(stage="screening", run_id="r", model="m",
                example_id="e1", condition="c", est_cost_usd=4.0,
                est_input_tokens=1, max_output_tokens=1)
    with pytest.raises(budget_mod.BudgetExceeded) as ei:
        led.reserve(stage="controls", run_id="r", model="m",
                    example_id="e2", condition="c", est_cost_usd=2.0,
                    est_input_tokens=1, max_output_tokens=1)
    assert "preliminary" in str(ei.value)


def test_ledger_total_cap_blocks_across_buckets(tmp_path):
    led = budget_mod.BudgetLedger(_budget_cfg(tmp_path))
    # 'evaluation' bucket has no sub-cap; total cap must still bind.
    with pytest.raises(budget_mod.BudgetExceeded) as ei:
        led.reserve(stage="evaluation", run_id="r", model="m",
                    example_id="e", condition="c", est_cost_usd=21.0,
                    est_input_tokens=1, max_output_tokens=1)
    assert "total cap" in str(ei.value)


def test_unresolved_reservation_stays_counted(tmp_path):
    led = budget_mod.BudgetLedger(_budget_cfg(tmp_path))
    r = led.reserve(stage="screening", run_id="r", model="m",
                    example_id="e", condition="c", est_cost_usd=3.0,
                    est_input_tokens=1, max_output_tokens=1)
    led.mark_unresolved(r["reservation_id"], run_id="r", example_id="e",
                        condition="c", error="timeout")
    st = led.state()
    assert r["reservation_id"] in st["unresolved_reservation_ids"]
    assert st["open_reserved_usd"] == pytest.approx(3.0)
    # The open reservation still counts against the cap.
    assert led.would_exceed("screening", 2.5) is not None
    # Manual release frees it.
    led.void(r["reservation_id"], reason="reviewed")
    assert led.state()["open_reserved_usd"] == 0.0


def test_cost_helpers():
    cfg = {"pricing": {"input_usd_per_1m": 2.5,
                       "output_usd_per_1m": 10.0}}
    c = budget_mod.conservative_cost(cfg, 1000, 1024)
    assert c == pytest.approx(1000 * 2.5e-6 + 1024 * 10e-6)
    a = budget_mod.actual_cost(
        cfg, {"prompt_tokens": 500, "completion_tokens": 100})
    assert a == pytest.approx(500 * 2.5e-6 + 100 * 10e-6)


def test_estimate_input_tokens_is_conservative_order():
    msgs = [{"role": "system", "content": "x" * 300},
            {"role": "user", "content": "y" * 600}]
    est = backends.estimate_input_tokens(msgs)
    # chars/3 + overhead = 300 + 4*2 + 3 = 311; must exceed a ~4 chars/tok guess
    assert est == 300 + 8 + 3
    assert est > 900 / 4


# ==========================================================================
# Mocked OpenAI provider normalization
# ==========================================================================

def _fake_completion(content="REASONING:\nx\n\nFINAL_ANSWER: 5",
                     finish="stop", pt=100, ct=20, refusal=None):
    return SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-4o-2024-08-06",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, refusal=refusal),
            finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct,
                              total_tokens=pt + ct))


def test_openai_response_normalized_verbatim():
    raw_text = "REASONING:\nstep\n\n\nstep2  \n\nFINAL_ANSWER: 5"
    res = backends.normalize_openai_response(
        _fake_completion(content=raw_text), 1.234)
    assert res["raw_response"] == raw_text  # never stripped
    assert res["finish_reason"] == "stop"
    assert res["hit_output_limit"] is False
    assert res["provider"]["usage"]["completion_tokens"] == 20
    assert res["provider"]["request_id"] == "chatcmpl-test"


def test_openai_length_maps_to_hit_limit_and_none_content():
    res = backends.normalize_openai_response(
        _fake_completion(content=None, finish="length"), 0.5)
    assert res["raw_response"] == ""        # None -> "" (string contract)
    assert res["hit_output_limit"] is True


def test_openai_refusal_recorded():
    res = backends.normalize_openai_response(
        _fake_completion(content=None, refusal="cannot help"), 0.1)
    assert res["provider"]["refusal"] == "cannot help"


def test_openai_backend_generate_with_fake_client():
    class FakeChat:
        class completions:
            @staticmethod
            def create(**kw):
                assert kw["temperature"] == 0.0
                assert kw["max_completion_tokens"] == 1024
                assert kw["model"] == "gpt-4o-2024-08-06"
                return _fake_completion()
    client = SimpleNamespace(chat=FakeChat)
    be = backends.OpenAIBackend(
        {"backend": "openai", "model": "gpt-4o-2024-08-06",
         "temperature": 0.0}, client=client)
    be.prepare()  # must not import openai when client injected
    res = be.generate([{"role": "user", "content": "hi"}], 1024)
    assert res["raw_response"].endswith("FINAL_ANSWER: 5")


def test_classify_provider_error():
    class Rejected(Exception):
        status_code = 429
    assert backends.classify_provider_error(
        Rejected("rate limited")) == "definitive"

    class Timeout(Exception):
        pass
    Timeout.__name__ = "APITimeoutError"
    assert backends.classify_provider_error(Timeout("t")) == "ambiguous"

    class ServerErr(Exception):
        status_code = 500
    assert backends.classify_provider_error(ServerErr("s")) == "ambiguous"


def test_openai_config_loads_and_identity():
    cfg = backends.load_model_config(
        REPO_ROOT / "configs/model_gpt-4o-2024-08-06.json")
    assert cfg["backend"] == "openai"
    ident = backends.model_identity(cfg)
    assert ident["repo"] == "openai"
    assert ident["revision"] == "gpt-4o-2024-08-06"
    spec = backends.decoding_spec(cfg, 1024)
    assert spec["temperature"] == 0.0


# ==========================================================================
# Dry-run isolation (no creds, no openai import, nothing written)
# ==========================================================================

def _openai_cfg(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"backend": "openai",
                             "model": "gpt-4o-2024-08-06",
                             "temperature": 0.0,
                             "pricing": {"input_usd_per_1m": 2.5,
                                         "output_usd_per_1m": 10.0}}))
    return p


def _tmp_inputs2(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    rec = {"example_id": "gsm8k-train-00001", "dataset": "gsm8k",
           "split": "train", "source_id": None,
           "question": "What is 2+2?", "context": None, "options": None}
    (processed / "gsm8k_train.inputs.jsonl").write_text(
        json.dumps(rec) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "stage": "screening",
        "pools": {"gsm8k": {"split": "train",
                            "example_ids": ["gsm8k-train-00001"]}}}))
    return manifest, processed


def test_dry_run_openai_no_creds_no_writes(tmp_path):
    import subprocess
    manifest, processed = _tmp_inputs2(tmp_path)
    out_dir = tmp_path / "run_oai"
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    proc = subprocess.run(
        [sys.executable, "scripts/run_generation.py", "--dry-run",
         "--manifest", str(manifest), "--processed-dir", str(processed),
         "--conditions", "couplets",
         "--model-config", str(_openai_cfg(tmp_path)),
         "--output-dir", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "backend: openai" in proc.stdout
    assert "stage: screening" in proc.stdout
    assert "planned responses: 1" in proc.stdout
    assert not out_dir.exists()


# ==========================================================================
# Review pack + import
# ==========================================================================

def _synth_run(outputs_root: Path, tmp_path, raw=COUPLET_RAW):
    """A complete fake run dir + manifest + inputs for review tests."""
    run_dir = outputs_root / "run_fake"
    run_dir.mkdir(parents=True)
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    rec_in = {"example_id": "gsm8k-train-00001", "dataset": "gsm8k",
              "split": "train", "source_id": None,
              "question": "What is 2+2?", "context": None, "options": None}
    (processed / "gsm8k_train.inputs.jsonl").write_text(
        json.dumps(rec_in) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pools": {"gsm8k": {
        "split": "train", "example_ids": ["gsm8k-train-00001"]}}}))
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_name": "run_fake",
        "compatibility": {
            "model_repo": "m/repo", "model_revision": "deadbeef00",
            "manifest_path": str(manifest),
        }}))
    resp = {"example_id": "gsm8k-train-00001", "dataset": "gsm8k",
            "condition": "couplets", "run_id": "run_fake",
            "model": {"repo": "m/repo", "revision": "deadbeef00"},
            "raw_response": raw,
            **rp.extract_reasoning_fields(raw, True)}
    (run_dir / "responses.jsonl").write_text(json.dumps(resp) + "\n")
    return run_dir, processed


def _synth_prompts(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir(exist_ok=True)
    (d / "system.txt").write_text("sys")
    for name in study_defs.POETIC_DEVICES + study_defs.CONTROLS:
        (d / f"{name}.txt").write_text("$question_block " + name)
    (d / "couplets.txt").write_text("$question_block\nRULE-COUPLETS\n"
                                   "FINAL_ANSWER: <a> $answer_spec")
    return d


def test_review_pack_blank_then_import(tmp_path):
    outputs_root = tmp_path / "outputs"
    run_dir, processed = _synth_run(outputs_root, tmp_path)
    prompts = _synth_prompts(tmp_path)
    out_dir = tmp_path / "reviews"

    import subprocess
    proc = subprocess.run(
        [sys.executable, "scripts/prepare_review.py",
         "--run-dirs", str(run_dir), "--out-dir", str(out_dir),
         "--prompts-dir", str(prompts),
         "--processed-dir", str(processed)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    csv_path = out_dir / "review.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["adherence_label"] == "" and row["notes"] == ""
    assert row["device_rule_text"].startswith("$question_block")
    rfile = Path(row["response_file"])
    txt = rfile.read_text(encoding="utf-8")
    assert COUPLET_RAW in txt           # raw preserved in review file
    assert "RULE-COUPLETS" in txt       # device instructions included
    assert "DIAGNOSTIC" in txt          # line-numbered view included
    assert "gold" not in txt.lower()    # no answer material

    # Fill in the label and import.
    row["adherence_label"] = "pass"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pr.CSV_FIELDS)
        w.writeheader()
        w.writerow(row)
    rep = ir.import_review(csv_path, outputs_root)
    assert rep["errors"] == []
    cell = list(rep["summary"].values())[0]
    assert cell["pass"] == 1 and cell["labeled"] == 1
    assert cell["unlabeled"] == 0


def test_import_review_rejects_bad_rows(tmp_path):
    outputs_root = tmp_path / "outputs"
    run_dir, processed = _synth_run(outputs_root, tmp_path)
    prompts = _synth_prompts(tmp_path)
    out_dir = tmp_path / "reviews"
    import subprocess
    subprocess.run(
        [sys.executable, "scripts/prepare_review.py",
         "--run-dirs", str(run_dir), "--out-dir", str(out_dir),
         "--prompts-dir", str(prompts),
         "--processed-dir", str(processed)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    csv_path = out_dir / "review.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    good = dict(rows[0])
    dup = dict(rows[0]); dup["adherence_label"] = "pass"
    bad_label = dict(rows[0]); bad_label["adherence_label"] = "great"
    unknown = dict(rows[0]); unknown["response_id"] = "nope::x::y::z::w"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pr.CSV_FIELDS)
        w.writeheader()
        w.writerows([good, dup, bad_label, unknown])
    rep = ir.import_review(csv_path, outputs_root)
    errs = "\n".join(rep["errors"])
    assert "duplicate" in errs and "unknown" in errs and "invalid" in errs
    # unlabeled valid row still counted in denominator
    cell = list(rep["summary"].values())[0]
    assert cell["total"] == 3 and cell["unlabeled"] == 2


# ==========================================================================
# Freeze selection
# ==========================================================================

def _sel(tmp_path, devices):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"devices": devices}))
    return p


def test_validate_selection_reports_problems(tmp_path):
    sel = _sel(tmp_path, {})  # every device missing
    retained, excluded, problems = fs.validate_selection(
        json.loads(sel.read_text()))
    assert not retained and len(problems) == 8
    sel = _sel(tmp_path, {n: {"decision": "retain", "reason": "ok"}
                          for n in study_defs.POETIC_DEVICES})
    retained, excluded, problems = fs.validate_selection(
        json.loads(sel.read_text()))
    assert retained == study_defs.POETIC_DEVICES and not problems


def test_freeze_and_verify_roundtrip(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.txt").write_text("sys")
    for name in study_defs.POETIC_DEVICES + study_defs.CONTROLS:
        (prompts / f"{name}.txt").write_text(f"rule {name}")
    sel = _sel(tmp_path, {
        n: {"decision": "retain" if n != "meter" else "exclude",
            "reason": "reviewed"}
        for n in study_defs.POETIC_DEVICES})
    frozen = tmp_path / "frozen.json"
    monkeypatch.setattr(fs, "PROMPTS_DIR", prompts)
    assert fs.freeze(sel, frozen, allow_empty=False) == 0
    fr = json.loads(frozen.read_text())
    assert "meter" in fr["excluded_devices"]
    assert fr["final_conditions"] == (
        study_defs.CONTROLS + [n for n in study_defs.POETIC_DEVICES
                               if n != "meter"])
    assert fs.verify(frozen) == 0
    (prompts / "couplets.txt").write_text("EDITED")
    assert fs.verify(frozen) == 1


def test_freeze_refuses_zero_retained(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts2"
    prompts.mkdir()
    (prompts / "system.txt").write_text("s")
    for n in study_defs.POETIC_DEVICES + study_defs.CONTROLS:
        (prompts / f"{n}.txt").write_text(n)
    sel = _sel(tmp_path, {n: {"decision": "exclude", "reason": "noisy"}
                          for n in study_defs.POETIC_DEVICES})
    monkeypatch.setattr(fs, "PROMPTS_DIR", prompts)
    assert fs.freeze(sel, tmp_path / "f.json", allow_empty=False) == 1
    assert fs.freeze(sel, tmp_path / "f.json", allow_empty=True) == 0


# ==========================================================================
# Eval manifest builder (content disjointness, determinism)
# ==========================================================================

def _proc_dir(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    def w(name, recs):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in recs))
    g_test = [{"example_id": f"gsm8k-test-{i:05d}", "dataset": "gsm8k",
               "question": f"test q{i}", "context": None, "options": None}
              for i in range(10)]
    g_test[3]["question"] = "DEV OVERLAP"   # duplicate of a train item
    w("gsm8k_test.inputs.jsonl", g_test)
    w("gsm8k_train.inputs.jsonl", [
        {"example_id": "gsm8k-train-00000", "dataset": "gsm8k",
         "question": "DEV OVERLAP", "context": None, "options": None}])
    l_opts = {"A": "a", "B": "b", "C": "c", "D": "d"}
    l_test = [{"example_id": f"logiqa_v2-test-{i:05d}",
               "dataset": "logiqa_v2", "question": f"q{i}",
               "context": f"ctx{i}", "options": l_opts}
              for i in range(6)]
    l_test[1]["context"] = "DEVCTX"; l_test[1]["question"] = "DEVQ"
    w("logiqa_v2_test.inputs.jsonl", l_test)
    w("logiqa_v2_validation.inputs.jsonl", [
        {"example_id": "logiqa_v2-validation-00000",
         "dataset": "logiqa_v2", "question": "DEVQ",
         "context": "DEVCTX", "options": l_opts}])
    return d


def test_eval_manifest_excludes_dev_content_and_is_deterministic(tmp_path):
    d = _proc_dir(tmp_path)
    m1 = bem.build_eval_manifest(d, 4, 42, "e")
    m2 = bem.build_eval_manifest(d, 4, 42, "e")
    assert m1 == m2
    g_ids = m1["pools"]["gsm8k"]["example_ids"]
    l_ids = m1["pools"]["logiqa_v2"]["example_ids"]
    assert "gsm8k-test-00003" not in g_ids        # dev-content overlap
    assert "logiqa_v2-test-00001" not in l_ids    # dev-content overlap
    assert len(g_ids) == 4 and len(l_ids) == 4
    assert m1["pools"]["gsm8k"]["excluded_for_dev_pool_content_overlap"] == 1
    assert m1["stage"] == "evaluation"


def test_eval_manifest_shortfall_reported(tmp_path):
    d = _proc_dir(tmp_path)
    m = bem.build_eval_manifest(d, 999, 42, "e")
    assert m["problems"]


# ==========================================================================
# Export filename collisions across models
# ==========================================================================

def test_export_filenames_distinct_across_models():
    rec_a = {"example_id": "e", "dataset": "gsm8k", "condition": "couplets",
             "model": {"repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                       "revision": "aaaa1111"}}
    rec_b = dict(rec_a, model={"repo": "openai",
                               "revision": "gpt-4o-2024-08-06"})
    fa = f"{ex.model_tag(rec_a['model'], {})}__{rec_a['dataset']}__" \
         f"{rec_a['condition']}__{rec_a['example_id']}.txt"
    fb = f"{ex.model_tag(rec_b['model'], {})}__{rec_b['dataset']}__" \
         f"{rec_b['condition']}__{rec_b['example_id']}.txt"
    assert fa != fb


# ==========================================================================
# Reuse comparison
# ==========================================================================

def test_reuse_compare_matching_and_mismatch(tmp_path):
    prompts = _synth_prompts(tmp_path)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"pools": {}}))
    import check_reuse as cr
    model_cfg = {"repo_id": "r", "revision": "sha1"}
    cur = cr.current_config(manifest, model_cfg, prompts,
                            ["couplets"], 1024)
    stored = {"backend": "mlx_local", "model_repo": "r",
              "model_revision": "sha1", "max_new_tokens": 1024,
              "sampler": cur["sampler"], "chat_template_sha256": "x",
              "prompt_template_sha256":
                  dict(cur["prompt_template_sha256"])}
    assert cr.compare(stored, cur) == []
    stored["model_revision"] = "other"
    assert cr.compare(stored, cur)
