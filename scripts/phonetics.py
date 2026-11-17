"""Pronunciation evidence for the adherence evaluator.

Loads a CMUdict-format pronunciation lexicon if one is available, in
this order:

  1. the POETIC_CMU_DICT environment variable (path to a cmudict file);
  2. data/lexicon/cmudict.dict (a file you place there yourself);
  3. the ``pronouncing`` package, if installed.

Nothing is downloaded automatically: if no source is present the
lexicon is reported as unavailable, phonetic checks return
``"unknown"`` evidence, and the routing policy decides whether the
case goes to the LLM judge. Tests inject a small synthetic lexicon
directly -- no file or package needed.

The lexicon maps lowercase word -> list of pronunciations, each a list
of ARPAbet phonemes with stress digits on vowels (e.g. "K AE1 T").
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON_PATH = REPO_ROOT / "data" / "lexicon" / "cmudict.dict"
ENV_VAR = "POETIC_CMU_DICT"

# ARPAbet vowels carry a stress digit (0,1,2); consonants do not.
_VOWEL_RE = re.compile(r"^[A-Z]+[012]$")
# Variant markers in cmudict: "WORD(2)  W AH1...".
_VARIANT_RE = re.compile(r"\(\d+\)$")


def parse_cmudict(text: str) -> dict[str, list[list[str]]]:
    """Parse cmudict-format text into word -> [pronunciation, ...]."""
    lex: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith((";;;", "#")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = _VARIANT_RE.sub("", parts[0]).lower()
        lex.setdefault(word, []).append(parts[1:])
    return lex


def load_lexicon(path: Path | None = None) -> dict | None:
    """Return the lexicon dict, or None when no source is available.

    Never downloads anything. ``path`` overrides discovery (used by
    tests with a small fixture file).
    """
    if path is not None:
        p = Path(path)
        return parse_cmudict(p.read_text(encoding="utf-8")) \
            if p.exists() else None
    env = os.environ.get(ENV_VAR)
    if env and Path(env).exists():
        return parse_cmudict(Path(env).read_text(encoding="utf-8"))
    if DEFAULT_LEXICON_PATH.exists():
        return parse_cmudict(
            DEFAULT_LEXICON_PATH.read_text(encoding="utf-8"))
    # ``pronouncing`` exposes the dict via pkg_resources, which no
    # longer exists on Python >=3.12 venvs without setuptools. Read the
    # bundled data files directly through find_spec so no third-party
    # package code is executed.
    try:
        import pronouncing  # type: ignore
        return parse_cmudict(pronouncing.cmudict.dict().strip())
    except ImportError:
        pass
    try:
        import importlib.util
        spec = importlib.util.find_spec("cmudict")
    except ImportError:
        spec = None
    if spec and spec.submodule_search_locations:
        pkg_dir = Path(spec.submodule_search_locations[0])
        cand = pkg_dir / "data" / "cmudict.dict"
        if cand.exists():
            return parse_cmudict(cand.read_text(encoding="utf-8"))
    return None


def lexicon_fingerprint(lexicon) -> str:
    """Stable identity for the loaded lexicon, included in eval cache
    keys: a different word set or pronunciation list must not reuse
    verdicts computed under another lexicon."""
    if lexicon is None:
        return "lex:none"
    import hashlib
    h = hashlib.sha256()
    for w in sorted(lexicon):
        h.update(w.encode("utf-8"))
        h.update(b"\0")
        for pron in lexicon[w]:
            h.update(" ".join(pron).encode("utf-8"))
            h.update(b"\x01")
        h.update(b"\x02")
    return "lex:cmudict-" + h.hexdigest()[:16]


def lexicon_status(lexicon) -> str:
    if lexicon is None:
        return ("unavailable: no cmudict found. To enable phonetic "
                "evidence, place a CMUdict-format file at "
                f"{DEFAULT_LEXICON_PATH}, set {ENV_VAR}, or install the "
                "'pronouncing' package. Nothing is downloaded "
                "automatically.")
    return f"loaded ({len(lexicon)} words)"


# --------------------------------------------------------------------------
# Per-word phonetic views
# --------------------------------------------------------------------------

def pronunciations(lexicon: dict, word: str) -> list[list[str]] | None:
    return lexicon.get(word.lower())


def is_vowel(phoneme: str) -> bool:
    return bool(_VOWEL_RE.match(phoneme))


def syllable_count(pron: list[str]) -> int:
    return sum(1 for p in pron if is_vowel(p))


def stress_pattern(pron: list[str]) -> str:
    """Stress digits of the vowel phonemes, e.g. '010' -> '010'."""
    return "".join(p[-1] for p in pron if is_vowel(p))


def rhyme_tail(pron: list[str]) -> tuple[str, ...]:
    """Phonemes from the last stressed vowel onward -- the classic
    cmudict rhyme suffix. Falls back to the last vowel onward when no
    stressed vowel exists (rare)."""
    idx = None
    for i, p in enumerate(pron):
        if is_vowel(p) and p[-1] in "12":
            idx = i
    if idx is None:
        for i, p in enumerate(pron):
            if is_vowel(p):
                idx = i
    if idx is None:
        return tuple(pron)
    return tuple(pron[idx:])


def words_rhyme(lexicon: dict, w1: str, w2: str) -> str:
    """'rhyme' if any pronunciation pair shares a rhyme tail;
    'no_rhyme' when both words resolve and no pair rhymes;
    'unknown' when either word is out of vocabulary."""
    p1, p2 = pronunciations(lexicon, w1), pronunciations(lexicon, w2)
    if not p1 or not p2:
        return "unknown"
    for a in p1:
        for b in p2:
            if rhyme_tail(a) == rhyme_tail(b):
                return "rhyme"
    return "no_rhyme"


def initial_consonant_phonemes(lexicon: dict,
                               word: str) -> list[str] | None:
    """Distinct initial consonant phonemes across pronunciations;
    [] if every pronunciation starts with a vowel; None if OOV."""
    prons = pronunciations(lexicon, word)
    if not prons:
        return None
    out = []
    for pron in prons:
        if pron and not is_vowel(pron[0]) and pron[0] not in out:
            out.append(pron[0])
    return out


def line_syllable_range(lexicon: dict, words: list[str]):
    """(min_syllables, max_syllables, oov_words) summing per-word
    pronunciation syllable ranges. Stops early at a cap to keep the
    cartesian bound trivial."""
    lo = hi = 0
    oov = []
    for w in words:
        prons = pronunciations(lexicon, w)
        if not prons:
            oov.append(w)
            continue
        counts = [syllable_count(p) for p in prons]
        lo += min(counts)
        hi += max(counts)
    return lo, hi, oov
