"""LLM judge client for adherence evaluation.

Builds the exact judge input (device rule + rubric + reasoning view +
optional mechanical evidence), sends it through the OpenAI chat
completions endpoint of the PINNED judge model (a separate config from
the three generating models), and parses the structured JSON verdict.

The judge NEVER receives: generating-model identity, gold answers,
answer-correctness results, or human labels. Question context is not
sent for any of the five frozen devices (see evaluator_policy.json).

Processing statuses (never silently converted into verdicts):
  resolved                    -- valid JSON verdict parsed
  judge_output_invalid        -- response text was not a valid verdict
  judge_output_truncated      -- finish_reason 'length'
  judge_request_error         -- definitive provider error (e.g. 4xx)
  judge_unresolved            -- ambiguous error; may have been billed,
                                 reservation stays open for review
  judge_skipped_too_long      -- reasoning view exceeded the input budget
"""

from __future__ import annotations

import json
import os
import string
import time
from pathlib import Path

import backends
import common

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = REPO_ROOT / "judge"
DEFAULT_JUDGE_CONFIG = (
    REPO_ROOT / "configs" / "judge_gpt-5-mini-2025-08-07.json")

VERDICT_VALUES = ("pass", "fail", "uncertain", "not_evaluable")

# Deliberate cap on judge input size: the evaluated reasoning is never
# truncated to save tokens; oversized views are flagged instead.
DEFAULT_MAX_INPUT_TOKENS_EST = 6000


def load_judge_templates(judge_dir: Path = JUDGE_DIR) -> dict:
    return {
        "system": (judge_dir / "system.txt").read_text(encoding="utf-8"),
        "user": string.Template(
            (judge_dir / "user.txt").read_text(encoding="utf-8")),
    }


def build_judge_messages(rule_text: str, rubric_text: str,
                         reasoning_text: str, evidence: str | None,
                         judge_dir: Path = JUDGE_DIR) -> list[dict]:
    """Exact messages sent to the judge. Contains ONLY the device rule,
    the rubric, optional mechanical evidence, and the reasoning text."""
    tpl = load_judge_templates(judge_dir)
    evidence_block = ""
    if evidence:
        evidence_block = (
            "MECHANICAL/PHONETIC EVIDENCE (advisory, may be incomplete):\n"
            + evidence)
    user = tpl["user"].substitute(
        rule_text=rule_text.strip(),
        rubric_text=rubric_text.strip(),
        evidence_block=evidence_block,
        reasoning_text=reasoning_text,
    )
    return [
        {"role": "system", "content": tpl["system"].strip()},
        {"role": "user", "content": user},
    ]


def parse_verdict(text: str) -> tuple[str, dict | None]:
    """Strict parse of the judge's JSON object.

    Returns ("resolved", verdict_dict) or ("judge_output_invalid", None).
    Extra keys are kept; required keys must exist and verdict must be one
    of VERDICT_VALUES.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        # Tolerate a markdown fence only by stripping the fence lines.
        lines = [l for l in s.splitlines() if not l.startswith("```")]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return "judge_output_invalid", None
    if not isinstance(obj, dict):
        return "judge_output_invalid", None
    if obj.get("verdict") not in VERDICT_VALUES:
        return "judge_output_invalid", None
    out = {
        "verdict": obj["verdict"],
        "units_evaluated": obj.get("units_evaluated"),
        "units_compliant": obj.get("units_compliant"),
        "evidence": obj.get("evidence"),
        "uncertainties": obj.get("uncertainties") or [],
    }
    extra = {k: v for k, v in obj.items() if k not in out}
    if extra:
        out["extra_fields"] = extra
    return "resolved", out


class JudgeClient:
    """Thin wrapper over a pinned OpenAI judge model.

    ``client`` may be injected for tests (SimpleNamespace fake) -- the
    ``openai`` package is only imported when a real client is needed.
    """

    def __init__(self, judge_cfg: dict, client=None):
        self.cfg = judge_cfg
        self._client = client

    @property
    def model(self) -> str:
        return self.cfg["model"]

    def prepare(self):
        if self._client is not None:
            return
        from openai import OpenAI  # lazy: absent in the test venv
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; export it before running "
                "the LLM judge (never hard-code or print it).")
        self._client = OpenAI(api_key=api_key, max_retries=0,
                              timeout=120.0)

    def estimate_input_tokens(self, messages: list[dict]) -> int:
        return backends.estimate_input_tokens(messages)

    def request_params(self) -> dict:
        params = {
            "model": self.cfg["model"],
            "max_completion_tokens": self.cfg.get("max_output_tokens", 600),
        }
        if self.cfg.get("temperature") is not None:
            params["temperature"] = self.cfg["temperature"]
        if self.cfg.get("response_format"):
            params["response_format"] = self.cfg["response_format"]
        return params

    def judge(self, messages: list[dict]) -> dict:
        """One judged call. Returns {raw_output, finish_reason, usage,
        request_id, provider, elapsed_seconds}."""
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            messages=messages, **self.request_params())
        elapsed = time.perf_counter() - t0
        norm = backends.normalize_openai_response(resp, elapsed)
        return {
            "raw_output": norm["raw_response"],
            "finish_reason": norm["finish_reason"],
            "usage": norm["provider"]["usage"],
            "request_id": norm["provider"]["request_id"],
            "model_returned": norm["provider"].get("model_returned"),
            "elapsed_seconds": norm["elapsed_seconds"],
        }
