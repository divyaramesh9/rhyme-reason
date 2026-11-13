"""Prompt construction for generation conditions.

Condition-specific instruction text lives in prompts/<condition>.txt so new
conditions (rhyme scheme, line breaks, syllable counts, ...) can be added by
dropping in a new template file without touching code. The shared system
message lives in prompts/system.txt.

Question formatting is identical across all conditions: templates receive an
already-built ``question_block`` (question, plus passage and options for
LogiQA) and an ``answer_spec`` describing the expected final-answer shape.

Only model-input fields (example_id, dataset, split, question, context,
options) may reach this module -- answers and metadata never do.
"""

from __future__ import annotations

import string
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = REPO_ROOT / "prompts"
SYSTEM_TEMPLATE_NAME = "system.txt"

OPTION_LABELS = ("A", "B", "C", "D")

ANSWER_SPECS = {
    "gsm8k": "the numeric answer to the problem (a number only)",
    "logiqa_v2": "one letter: A, B, C, or D",
}


def available_conditions(prompt_dir: Path = PROMPT_DIR) -> list[str]:
    """Condition names = prompt template files, minus the shared system file."""
    return sorted(
        p.stem for p in prompt_dir.glob("*.txt") if p.name != SYSTEM_TEMPLATE_NAME
    )


def load_templates(prompt_dir: Path = PROMPT_DIR) -> dict:
    """Load the shared system message and every condition template.

    Returns {"system": str, "conditions": {name: string.Template}}.
    """
    system = (prompt_dir / SYSTEM_TEMPLATE_NAME).read_text(encoding="utf-8")
    conditions = {
        name: string.Template((prompt_dir / f"{name}.txt").read_text(
            encoding="utf-8"))
        for name in available_conditions(prompt_dir)
    }
    if not conditions:
        raise FileNotFoundError(f"no condition templates found in {prompt_dir}")
    return {"system": system, "conditions": conditions}


def question_block(record: dict) -> str:
    """Identical question formatting for every condition and dataset."""
    if record["dataset"] == "logiqa_v2":
        options = record["options"] or {}
        lines = [
            f"{label}. {options[label]}"
            for label in OPTION_LABELS
            if label in options
        ]
        return (
            "Passage:\n"
            f"{record['context']}\n\n"
            "Question:\n"
            f"{record['question']}\n\n"
            "Options:\n" + "\n".join(lines)
        )
    return f"Question:\n{record['question']}"


def answer_spec(dataset: str) -> str:
    if dataset not in ANSWER_SPECS:
        raise KeyError(f"no answer spec for dataset {dataset!r}")
    return ANSWER_SPECS[dataset]


def build_user_message(record: dict, condition: str, templates: dict) -> str:
    """Render the user message for one example under one condition."""
    try:
        template = templates["conditions"][condition]
    except KeyError:
        raise KeyError(
            f"unknown condition {condition!r}; "
            f"available: {sorted(templates['conditions'])}"
        ) from None
    return template.substitute(
        question_block=question_block(record),
        answer_spec=answer_spec(record["dataset"]),
    )


def build_messages(record: dict, condition: str, templates: dict) -> list[dict]:
    """Fresh conversation: shared system message + condition user message."""
    return [
        {"role": "system", "content": templates["system"].strip()},
        {"role": "user", "content": build_user_message(record, condition,
                                                       templates)},
    ]
