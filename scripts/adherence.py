"""Deterministic adherence evaluation for the five frozen devices.

This module turns a saved response record into a structured verdict for
one poetic device. It implements only what can be checked reliably in
code; the routing policy (configs/evaluator_policy.json) decides which
unresolved cases go to the LLM judge. The operational rubric -- which
lines are evaluated, case/whitespace/punctuation policy, and every
documented ambiguity -- lives in docs/adherence_rubric.md. Nothing here
normalizes the stored record: evaluation works on a derived "reasoning
view" and leaves raw_response / reasoning_text untouched.

Verdicts: "pass" | "fail" | "uncertain" | "not_evaluable".
Processing statuses (distinct from verdicts):
  resolved                -- a verdict was produced
  no_applicable_lines     -- the reasoning view has no non-blank lines
  missing_reasoning_view  -- ambiguous markers / no usable view
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import common
import phonetics as ph
import response_parsing as rp

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
RUBRIC_DIR = REPO_ROOT / "judge" / "rubrics"

EVALUATOR_VERSION = "1.1.1"
VERDICTS = ("pass", "fail", "uncertain", "not_evaluable")

# Device constants -- the supplied phrases are FROZEN (prompts/anaphora.txt
# and prompts/epistrophe.txt); changing them means a new frozen rule.
ANAPHORA_PHRASE = "We know "
EPISTROPHE_PHRASE = "as we see"

CONSONANT_LETTERS = frozenset("bcdfghjklmnpqrstvwxz")
# End-of-line policy (user decision, rubric Q2): 'as we see' must be
# the last WORDS on the line. Any reasonable punctuation may follow it
# -- 'as we see,' is fine, as in normal verse -- but no words, digits,
# or non-punctuation symbols may come after the phrase.
TRAILING_PUNCT = ".,;:!?\u2026-–—\"'\u201c\u201d\u2018\u2019)]}"

FROZEN_DEVICES = ("anaphora", "epistrophe", "alliteration",
                  "rhyme_scheme", "meter")


# --------------------------------------------------------------------------
# Derived reasoning view + line policy
# --------------------------------------------------------------------------

def reasoning_view(rec: dict) -> tuple[str | None, str]:
    """The text the evaluator looks at. Returns (text, source).

    Order: the stored whitespace-preserving reasoning_text when present;
    otherwise, when the REASONING: marker is simply missing, the raw
    response up to the first FINAL_ANSWER: marker (the human reviewers
    judged exactly this visible text). Ambiguous markers or answer-only
    conditions yield (None, status).
    """
    rt = rec.get("reasoning_text")
    if rt is not None:
        return rt, "reasoning_text"
    status = rec.get("reasoning_parse_status")
    if status in (rp.STATUS_AMBIGUOUS, rp.STATUS_NOT_APPLICABLE):
        return None, status
    raw = rp.record_raw_text(rec)
    m = rp.FINAL_ANSWER_LINE_RE.search(raw)
    view = raw[:m.start()] if m else raw
    # A stray REASONING: label line is not a reasoning line.
    view = rp.REASONING_RE.sub("", view, count=1)
    return view, "raw_minus_final_answer"


def reasoning_lines(view: str) -> list[dict]:
    """Applicable reasoning lines: non-blank lines, in order.

    Each entry: {"n": 1-based line number within the view, "text":
    original line, "stripped": whitespace-stripped line}. Whitespace-only
    lines are separators/formatting, never evaluated units. A trailing
    CR is removed from the checked text only.
    """
    out = []
    for i, line in enumerate(view.split("\n"), 1):
        checked = line.rstrip("\r")
        if checked.strip():
            out.append({"n": i, "text": line, "stripped": checked.strip()})
    return out


def _no_lines(device: str, view_source: str) -> dict:
    return {
        "device": device,
        "verdict": "not_evaluable",
        "processing_status": "no_applicable_lines",
        "reasoning_view_source": view_source,
        "units": [],
        "unit_counts": {"applicable": 0, "passed": 0, "failed": 0,
                        "unresolved": 0, "exempt": 0},
        "evidence_summary": "no non-blank reasoning lines to evaluate",
        "needs_judge": False,
    }


def _aggregate(units: list[dict], device: str, view: str,
               view_source: str, notes: list[str]) -> dict:
    if not units:
        # No vacuous pass: nothing to evaluate is not_evaluable.
        out = _no_lines(device, view_source)
        out["reasoning_view_source"] = view_source
        return out
    failed = [u for u in units if u["result"] == "fail"]
    unresolved = [u for u in units if u["result"] == "unknown"]
    applicable = [u for u in units if u["result"] != "exempt"]
    if failed:
        verdict = "fail"
    elif unresolved:
        verdict = "uncertain"
    else:
        verdict = "pass"
    return {
        "device": device,
        "verdict": verdict,
        "processing_status": "resolved",
        "reasoning_view_source": view_source,
        "units": units,
        "unit_counts": {
            "applicable": len(applicable),
            "passed": sum(1 for u in units if u["result"] == "pass"),
            "failed": len(failed),
            "unresolved": len(unresolved),
            "exempt": sum(1 for u in units if u["result"] == "exempt"),
        },
        "evidence_summary": "; ".join(notes) if notes else "",
        "needs_judge": bool(unresolved),
    }


# --------------------------------------------------------------------------
# anaphora / epistrophe -- literal supplied-phrase checks
# --------------------------------------------------------------------------

def eval_anaphora(view: str, view_source: str) -> dict:
    """Every applicable line must begin with the phrase 'We know '.

    Policy (docs/adherence_rubric.md): case-sensitive; leading
    whitespace ignored; the phrase must be followed by a word boundary
    (so 'We known...' fails); a line that contains ONLY the phrase
    (e.g. 'We know', 'We know.') is a FAIL -- it carries no reasoning
    content (user decision, rubric Q1).
    """
    units, notes = [], []
    for ln in reasoning_lines(view):
        s = ln["stripped"]
        m = re.match(r"^We know\b", s)
        ok = bool(m)
        detail = f"line {ln['n']}: " + (
            "begins with the supplied phrase" if ok
            else f"does not begin with 'We know ': {s[:60]!r}")
        if ok:
            rest = s[m.end():].strip().strip(string.punctuation).strip()
            if not rest:
                ok = False
                detail += " [phrase-only line: no reasoning content]"
                notes.append(
                    f"line {ln['n']} contains only the phrase (fail)")
        units.append({"unit": f"line {ln['n']}", "text": ln["text"],
                      "result": "pass" if ok else "fail",
                      "detail": detail})
    return _aggregate(units, "anaphora", view, view_source, notes)


def eval_epistrophe(view: str, view_source: str) -> dict:
    """Every applicable line must end with the phrase 'as we see'.

    Policy (user decision, rubric Q2): the phrase must be the last
    WORDS on the line -- any reasonable punctuation may follow it
    ('as we see.'/'as we see,' both count, as in normal verse), but
    words, digits, or stray symbols after the phrase fail. Trailing
    whitespace ignored; the phrase must start on a word boundary
    ('...ideas we see' does NOT count).
    """
    units, notes = [], []
    for ln in reasoning_lines(view):
        s = ln["text"].rstrip("\r").rstrip()
        strict = s.endswith(EPISTROPHE_PHRASE)
        stripped = s.rstrip(TRAILING_PUNCT).rstrip()
        tolerant = stripped.endswith(EPISTROPHE_PHRASE)
        boundary = tolerant and (
            len(stripped) == len(EPISTROPHE_PHRASE)
            or not stripped[-len(EPISTROPHE_PHRASE) - 1].isalpha())
        ok = boundary
        detail = f"line {ln['n']}: " + (
            "ends with the supplied phrase" if ok
            else f"does not end with 'as we see': {s[-60:]!r}")
        if ok and not strict:
            detail += " [trailing punctuation tolerated]"
            notes.append(f"line {ln['n']}: punctuation after phrase")
        units.append({"unit": f"line {ln['n']}", "text": ln["text"],
                      "result": "pass" if ok else "fail",
                      "detail": detail})
    return _aggregate(units, "epistrophe", view, view_source, notes)


# --------------------------------------------------------------------------
# alliteration -- the letter rule, plus phonetic evidence
# --------------------------------------------------------------------------

def _line_words(stripped: str) -> list[str]:
    """Whitespace tokens reduced to word characters for checking:
    edge punctuation stripped; a token counts only if a letter remains."""
    words = []
    for tok in stripped.split():
        w = tok.strip(string.punctuation)
        if any(c.isalpha() for c in w):
            words.append(w)
    return words


def eval_alliteration(view: str, view_source: str,
                      lexicon=None) -> dict:
    """Per line: >=3 words starting with the same consonant letter
    (the rule as frozen). Phonetic initial-sound agreement is attached
    as evidence when a lexicon is available -- initial letters alone are
    not treated as proof of initial-sound repetition."""
    units, notes = [], []
    for ln in reasoning_lines(view):
        words = _line_words(ln["stripped"])
        counts: dict[str, list[str]] = {}
        for w in words:
            first = next((c for c in w if c.isalpha()), "")
            if first and first.lower() in CONSONANT_LETTERS:
                counts.setdefault(first.lower(), []).append(w)
        letter, members = ("", [])
        if counts:
            letter, members = max(counts.items(), key=lambda kv: len(kv[1]))
        ok = len(members) >= 3
        detail = (f"line {ln['n']}: {len(members)} word(s) start with "
                  f"consonant '{letter}' {members[:6]}")
        unit = {"unit": f"line {ln['n']}", "text": ln["text"],
                "result": "pass" if ok else "fail", "detail": detail,
                "letter_check": {"letter": letter, "count": len(members),
                                 "words": members}}
        if lexicon is not None:
            sound = _alliteration_sound_check(lexicon, members)
            unit["phonetic_evidence"] = sound
            if ok and sound["agreement"] is False:
                unit["detail"] += (
                    " [phonetic: letter match but initial SOUNDS differ "
                    "-- judge review advised]")
                notes.append(f"line {ln['n']}: letter/sound mismatch "
                             f"({sound['detail']})")
        else:
            unit["phonetic_evidence"] = {"agreement": None,
                                         "detail": "lexicon unavailable"}
        units.append(unit)
    return _aggregate(units, "alliteration", view, view_source, notes)


def _alliteration_sound_check(lexicon: dict, members: list[str]) -> dict:
    """Do the letter-matched words also share an initial consonant
    PHONEME under at least one pronunciation each?"""
    seen, oov = [], []
    for w in members:
        ps = ph.initial_consonant_phonemes(lexicon, w)
        (oov if ps is None else seen).append((w, ps))
    if oov:
        return {"agreement": None,
                "detail": f"out-of-vocabulary: {oov}"}
    shared = set(seen[0][1]) if seen else set()
    for _, ps in seen[1:]:
        shared &= set(ps)
    if shared:
        return {"agreement": True,
                "detail": f"shared initial phoneme(s) {sorted(shared)}"}
    return {"agreement": False,
            "detail": "no common initial phoneme across pronunciations"}


# --------------------------------------------------------------------------
# rhyme_scheme -- AABB pair rhyme on the last words
# --------------------------------------------------------------------------

def _last_word(stripped: str) -> str | None:
    """Last whitespace token, edge punctuation stripped, lowercased."""
    words = _line_words(stripped)
    return words[-1].lower() if words else None


def eval_rhyme_scheme(view: str, view_source: str,
                      lexicon=None) -> dict:
    """Consecutive applicable lines paired (1,2), (3,4), ... must end in
    rhyming last words that are DIFFERENT words; an odd trailing line is
    exempt. Rhyme = shared rhyme tail (last stressed vowel onward) under
    any listed pronunciation; OOV pairs are unresolved, not failed."""
    lines = reasoning_lines(view)
    units, notes = [], []
    for i in range(0, len(lines), 2):
        a = lines[i]
        if i + 1 >= len(lines):
            units.append({"unit": f"line {a['n']}", "text": a["text"],
                          "result": "exempt",
                          "detail": f"line {a['n']}: odd trailing line "
                                    "(exempt per the frozen rule)"})
            continue
        b = lines[i + 1]
        wa, wb = _last_word(a["stripped"]), _last_word(b["stripped"])
        tag = f"lines {a['n']}&{b['n']}"
        unit = {"unit": tag, "texts": [a["text"], b["text"]],
                "end_words": [wa, wb]}
        if wa is None or wb is None:
            unit.update(result="unknown",
                        detail=f"{tag}: cannot identify an end word")
        elif wa == wb:
            unit.update(
                result="unknown",
                detail=(f"{tag}: identical end word {wa!r} -- repetition "
                        "may be genuine or just for the rhyme; judge "
                        "decides (user decision, rubric Q3)"))
        elif lexicon is None:
            unit.update(result="unknown",
                        detail=f"{tag}: {wa!r}/{wb!r} rhyme check needs "
                               "a lexicon (unavailable)")
        else:
            r = ph.words_rhyme(lexicon, wa, wb)
            if r == "rhyme":
                unit.update(result="pass",
                            detail=f"{tag}: {wa!r}/{wb!r} share a rhyme "
                                   "tail")
            elif r == "no_rhyme":
                unit.update(
                    result="unknown",
                    detail=(f"{tag}: {wa!r}/{wb!r} do not share a "
                            "strict rhyme tail -- a slant rhyme may "
                            "still count; judge decides (user "
                            "decision, rubric Q3)"))
            else:
                unit.update(result="unknown",
                            detail=f"{tag}: pronunciation unknown for "
                                   f"{wa!r}/{wb!r}")
        units.append(unit)
    if lexicon is None:
        notes.append("pronunciation lexicon unavailable; pairs unresolved")
    res = _aggregate(units, "rhyme_scheme", view, view_source, notes)
    if len(lines) < 2:
        # Fewer than one complete pair: the AABB rule cannot be
        # demonstrated at all, and vacuously passing a one-line response
        # contradicts the observed review convention. Judge decides.
        res["verdict"] = "uncertain"
        res["processing_status"] = "resolved"
        notes.append("fewer than 2 applicable lines: no rhyme pair "
                     "formed; routed to judge")
        res["evidence_summary"] = "; ".join(notes)
    res["needs_judge"] = (res["verdict"] == "uncertain")
    # Whether the end words are "part of the reasoning sentences" is a
    # semantic check code cannot make -- always routed to the judge when
    # a pair is otherwise resolved but suspiciously short, per rubric.
    res["needs_judge"] = res["needs_judge"] or any(
        u["result"] == "unknown" for u in units)
    return res


# --------------------------------------------------------------------------
# meter -- evidence only; code never issues a pass
# --------------------------------------------------------------------------

def eval_meter(view: str, view_source: str, lexicon=None) -> dict:
    """Iambic pentameter: an unstressed-stressed pattern over ~10
    syllables per line. Deterministic code can only FAIL a line whose
    words cannot sum to 10 syllables under any listed pronunciation;
    everything else is 'unknown' evidence for the judge (lexical stress
    in context is genuinely ambiguous -- see the rubric)."""
    units, notes = [], []
    for ln in reasoning_lines(view):
        words = _line_words(ln["stripped"])
        unit = {"unit": f"line {ln['n']}", "text": ln["text"]}
        if lexicon is None:
            unit.update(result="unknown",
                        detail=f"line {ln['n']}: no lexicon -- syllable/"
                               "stress evidence unavailable")
        else:
            lo, hi, oov = ph.line_syllable_range(lexicon, words)
            if oov:
                unit.update(result="unknown",
                            detail=(f"line {ln['n']}: out-of-vocabulary "
                                    f"{oov[:6]}; syllable range over "
                                    f"known words {lo}-{hi}"))
            elif not (lo <= 10 <= hi):
                unit.update(result="fail",
                            detail=(f"line {ln['n']}: {lo}-{hi} syllables "
                                    "possible -- cannot reach the "
                                    "10-syllable iambic pentameter line "
                                    "under any listed pronunciation"))
            else:
                unit.update(
                    result="unknown",
                    detail=(f"line {ln['n']}: {lo}-{hi} syllables possible "
                            "(10 reachable); stress pattern ambiguous "
                            "-- judge decides"))
        units.append(unit)
    if lexicon is None:
        notes.append("pronunciation lexicon unavailable; lines unresolved")
    res = _aggregate(units, "meter", view, view_source, notes)
    res["needs_judge"] = res["verdict"] != "fail" or bool(
        res["unit_counts"]["unresolved"])
    return res


DEVICE_EVALUATORS = {
    "anaphora": eval_anaphora,
    "epistrophe": eval_epistrophe,
    "alliteration": eval_alliteration,
    "rhyme_scheme": eval_rhyme_scheme,
    "meter": eval_meter,
}


# --------------------------------------------------------------------------
# Record-level evaluation + eval-key computation
# --------------------------------------------------------------------------

def rule_sha256(condition: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    return common.sha256_file(prompts_dir / f"{condition}.txt")


def rubric_sha256(condition: str, rubric_dir: Path = RUBRIC_DIR) -> str | None:
    p = rubric_dir / f"{condition}.txt"
    return common.sha256_file(p) if p.exists() else None


def evaluate(rec: dict, lexicon=None,
             prompts_dir: Path = PROMPTS_DIR,
             rubric_dir: Path = RUBRIC_DIR) -> dict:
    """Deterministic verdict payload for one response record."""
    device = rec["condition"]
    fn = DEVICE_EVALUATORS[device]
    view, view_source = reasoning_view(rec)
    if view is None:
        out = _no_lines(device, view_source or "none")
        out["processing_status"] = "missing_reasoning_view"
        return out
    if device in ("anaphora", "epistrophe"):
        return fn(view, view_source)
    return fn(view, view_source, lexicon)


def eval_key(response_id: str, rec: dict, source: str,
             rubric_sha: str | None, policy_sha: str,
             judge_model: str | None) -> str:
    """Cache/resume key: response identity + content + every version
    that could change the verdict."""
    parts = [
        response_id,
        common.sha256_text(rp.record_raw_text(rec)),
        rec["condition"],
        rule_sha256(rec["condition"]),
        rubric_sha or "no-rubric",
        EVALUATOR_VERSION,
        policy_sha,
        judge_model or "no-judge",
        source,
    ]
    return common.sha256_text("|".join(parts))
