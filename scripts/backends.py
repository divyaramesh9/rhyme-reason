"""Provider backends for generation.

Two backends share one interface so run_generation.py stays provider-
agnostic:

    backend = make_backend(model_cfg)
    backend.prepare()                      # load/resolve whatever it needs
    result = backend.generate(messages, max_new_tokens)
    info   = backend.compatibility_info()  # identity + decoding + template sha

* ``mlx_local`` -- local MLX-LM inference (Qwen, Llama, ...), greedy
  decoding. Weights are resolved through the pinned-revision
  snapshot_download path the project already uses.
* ``openai``    -- OpenAI Chat Completions for a dated model snapshot.
  The ``openai`` package is imported lazily so dry runs and the test
  suite never need it installed or credentials present.

Both return the same result dict; provider-specific details live under
``provider``. The raw response text is whatever the backend surfaced --
it is never stripped or normalized here.
"""

from __future__ import annotations

import math
import os
import platform
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MLX_SAMPLER_SPEC = {
    "name": "greedy",
    "implementation": "mlx_lm.sample_utils.make_sampler(temp=0.0) -> argmax",
    "temperature": 0.0,
    "repetition_penalty": None,
    "top_p": None,
    "top_k": None,
}

# Offline input-token estimate for OpenAI calls. The o200k tokenizer is
# not guaranteed available offline, so we use a deliberately conservative
# character heuristic: English OpenAI tokenization averages ~4 chars/token
# (often better), so chars/3 overestimates by ~30%. Per-message overhead
# (~4 tokens) plus the assistant priming (~3) follows the published
# chat-completions accounting guidance.
CHARS_PER_TOKEN_CONSERVATIVE = 3.0
PER_MESSAGE_OVERHEAD_TOKENS = 4
REPLY_PRIMING_TOKENS = 3


def estimate_input_tokens(messages: list[dict]) -> int:
    """Conservative, approximate input-token estimate (labeled as such)."""
    chars = sum(len(m.get("content") or "") for m in messages)
    return (math.ceil(chars / CHARS_PER_TOKEN_CONSERVATIVE)
            + PER_MESSAGE_OVERHEAD_TOKENS * len(messages)
            + REPLY_PRIMING_TOKENS)


def load_model_config(path: Path) -> dict:
    """Load a model config JSON and normalize it to a backend spec."""
    import json
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("backend", "mlx_local")
    if cfg["backend"] == "openai":
        cfg.setdefault("temperature", 0.0)
    return cfg


def model_identity(cfg: dict) -> dict:
    """model dict recorded in run_config and every response record."""
    if cfg.get("backend") == "openai":
        return {
            "repo": "openai",
            "revision": cfg["model"],
            "quantization": None,
            "backend": "openai",
        }
    return {
        "repo": cfg["repo_id"],
        "revision": cfg["revision"],
        "quantization": None,  # filled in after snapshot resolution
        "backend": "mlx_local",
    }


def decoding_spec(cfg: dict, max_new_tokens: int) -> dict:
    """Decoding settings recorded for compatibility and provenance."""
    if cfg.get("backend") == "openai":
        return {
            "name": "openai_chat_completions",
            "temperature": cfg.get("temperature", 0.0),
            "max_completion_tokens": max_new_tokens,
            "note": ("temperature=0 requested for reproducibility; OpenAI "
                     "does not guarantee identical repeated outputs."),
        }
    return dict(MLX_SAMPLER_SPEC)


def classify_provider_error(exc: BaseException) -> str:
    """'ambiguous' (the request may have reached the model and been
    billed -- e.g. timeouts, connection drops, 5xx) or 'definitive'
    (clearly rejected before generation, e.g. 4xx). Classification uses
    exception attributes/type names so it also works on mocked errors."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return "ambiguous" if status >= 500 else "definitive"
    name = type(exc).__name__.lower()
    if any(k in name for k in ("timeout", "connection", "ratelimit")):
        return "ambiguous"
    return "ambiguous"


# --------------------------------------------------------------------------
# Local MLX backend
# --------------------------------------------------------------------------

class MlxLocalBackend:
    """Local inference through mlx_lm, unchanged from the existing
    pipeline: pinned snapshot, greedy sampler, one prompt at a time."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.snapshot_dir: Path | None = None
        self.snapshot_seconds = 0.0
        self.load_seconds = 0.0

    def prepare(self) -> None:
        from huggingface_hub import snapshot_download
        import mlx.core as mx
        from mlx_lm import load as mlx_load

        t0 = time.perf_counter()
        rev = self.cfg.get("revision")
        kwargs = {"revision": rev} if rev else {}
        self.snapshot_dir = Path(
            snapshot_download(self.cfg["repo_id"], **kwargs))
        self.snapshot_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.model, self.tokenizer = mlx_load(str(self.snapshot_dir))
        mx.eval(self.model.parameters())
        self.load_seconds = time.perf_counter() - t1
        self._mx = mx

    def resolved_revision(self) -> str | None:
        if self.snapshot_dir is not None:
            return self.snapshot_dir.name  # HF cache: snapshots/<sha>
        return self.cfg.get("revision")

    def chat_template_sha(self) -> str:
        import hashlib
        tpl = getattr(self.tokenizer, "chat_template", None)
        if isinstance(tpl, str):
            return hashlib.sha256(tpl.encode("utf-8")).hexdigest()
        return "unavailable"

    def quantization(self):
        import json
        if self.snapshot_dir is None:
            return None
        cfg_path = self.snapshot_dir / "config.json"
        if not cfg_path.exists():
            return None
        return json.loads(cfg_path.read_text()).get("quantization")

    def environment_info(self, memory_field_doc: dict) -> dict:
        import importlib.metadata as md

        def ver(p):
            try:
                return md.version(p)
            except md.PackageNotFoundError:
                return None

        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                p: ver(p)
                for p in ("mlx", "mlx-lm", "transformers",
                          "huggingface-hub", "tokenizers", "numpy")
                if ver(p) is not None
            },
            "model_config": self._model_config_info(),
            "chat_template_sha256": self.chat_template_sha(),
            "memory_field_meanings": memory_field_doc,
        }

    def _model_config_info(self) -> dict:
        import json
        cfg_path = self.snapshot_dir / "config.json"
        model_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() \
            else {}
        return {
            "model_type": model_cfg.get("model_type"),
            "quantization": model_cfg.get("quantization"),
            "torch_dtype": model_cfg.get("torch_dtype"),
        }

    def generate(self, messages: list[dict], max_new_tokens: int) -> dict:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx = self._mx
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        try:
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            elif hasattr(mx.metal, "reset_peak_memory"):
                mx.metal.reset_peak_memory()
        except Exception:
            pass

        t0 = time.perf_counter()
        parts, prompt_tokens, gen_tokens = [], None, None
        peak_gib = finish_reason = gen_tps = prompt_tps = None
        for r in stream_generate(
                self.model, self.tokenizer, prompt=prompt_text,
                max_tokens=max_new_tokens,
                sampler=make_sampler(temp=0.0)):
            parts.append(r.text)
            prompt_tokens = r.prompt_tokens
            gen_tokens = r.generation_tokens
            peak_gib = r.peak_memory
            gen_tps = r.generation_tps
            prompt_tps = r.prompt_tps
            if r.finish_reason:
                finish_reason = r.finish_reason
        elapsed = time.perf_counter() - t0

        return {
            "prompt": prompt_text,
            "raw_response": "".join(parts),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "prompt_tps": prompt_tps,
            "generation_tps": gen_tps,
            "elapsed_seconds": round(elapsed, 3),
            "finish_reason": finish_reason,
            "hit_output_limit": finish_reason == "length",
            "memory": memory_snapshot(mx, peak_gib),
            "provider": None,
        }


def memory_snapshot(mx, gen_response_peak_gib) -> dict:
    """Memory measurements; None where the API is unavailable."""
    import resource
    snap = {
        "generation_response_peak_gib": gen_response_peak_gib,
        "process_maxrss_bytes": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss,
    }
    try:
        def get(name):
            if hasattr(mx, name):
                return getattr(mx, name)
            return getattr(mx.metal, name, None)

        peak = get("get_peak_memory")
        active = get("get_active_memory")
        cache = get("get_cache_memory")
        snap["mlx_peak_memory_bytes"] = peak() if peak else None
        snap["mlx_active_memory_bytes"] = active() if active else None
        snap["mlx_cache_memory_bytes"] = cache() if cache else None
    except Exception:
        snap["mlx_peak_memory_bytes"] = None
        snap["mlx_active_memory_bytes"] = None
        snap["mlx_cache_memory_bytes"] = None
    return snap


# --------------------------------------------------------------------------
# OpenAI backend
# --------------------------------------------------------------------------

OPENAI_FINISH_MAP = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "function_call": "function_call",
    "tool_calls": "tool_calls",
}


class OpenAIBackend:
    """OpenAI Chat Completions backend for a dated model snapshot.

    Sequential, no SDK retries (max_retries=0): every billed attempt is
    deliberate. ``client`` may be injected for tests -- the ``openai``
    package is only imported when a real client is needed.
    """

    def __init__(self, cfg: dict, client=None):
        self.cfg = cfg
        self._client = client
        self.prepare_seconds = 0.0

    def prepare(self) -> None:
        if self._client is not None:
            return
        t0 = time.perf_counter()
        from openai import OpenAI  # lazy: absent in the test venv
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; export it before running "
                "OpenAI generation (never hard-code or print it).")
        self._client = OpenAI(api_key=api_key, max_retries=0, timeout=120.0)
        self.prepare_seconds = time.perf_counter() - t0

    def resolved_revision(self) -> str:
        return self.cfg["model"]

    def chat_template_sha(self) -> str:
        return "n/a (provider-side chat template)"

    def quantization(self):
        return None

    def environment_info(self, memory_field_doc: dict) -> dict:
        import importlib.metadata as md
        try:
            sdk = md.version("openai")
        except md.PackageNotFoundError:
            sdk = None
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {"openai": sdk} if sdk else {},
            "provider": "openai",
            "chat_template_sha256": self.chat_template_sha(),
            "model_config": {
                "model": self.cfg["model"],
                "quantization": None,
            },
            "memory_field_meanings": "n/a (remote provider)",
        }

    def generate(self, messages: list[dict], max_new_tokens: int) -> dict:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.cfg["model"],
            messages=messages,
            temperature=self.cfg.get("temperature", 0.0),
            max_completion_tokens=max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
        return normalize_openai_response(resp, elapsed)


def normalize_openai_response(resp, elapsed: float) -> dict:
    """Map a ChatCompletion(-like) object onto the shared result dict.

    Works on real SDK objects and on the SimpleNamespace fakes used in
    tests. The message content is taken verbatim -- None (refusals /
    content filter) becomes "" so downstream always sees a string.
    """
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content if getattr(msg, "content", None) else ""
    refusal = getattr(msg, "refusal", None) or None
    usage = getattr(resp, "usage", None)
    finish = getattr(choice, "finish_reason", None)
    provider = {
        "request_id": getattr(resp, "id", None),
        "model_returned": getattr(resp, "model", None),
        "finish_reason_raw": finish,
        "refusal": refusal,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }
    mapped = OPENAI_FINISH_MAP.get(finish, finish or "unknown")
    return {
        "prompt": None,
        "raw_response": content,
        "prompt_tokens": provider["usage"]["prompt_tokens"],
        "generated_tokens": provider["usage"]["completion_tokens"],
        "prompt_tps": None,
        "generation_tps": None,
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": mapped,
        "hit_output_limit": mapped == "length",
        "memory": {},
        "provider": provider,
    }


def make_backend(model_cfg: dict, client=None):
    """Factory: pick the backend named by the model config."""
    backend = model_cfg.get("backend", "mlx_local")
    if backend == "mlx_local":
        return MlxLocalBackend(model_cfg)
    if backend == "openai":
        return OpenAIBackend(model_cfg, client=client)
    raise ValueError(f"unknown backend {backend!r}")
