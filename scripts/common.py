"""Small shared helpers for the pipeline scripts.

Stdlib-only so every script (including the offline tools that run under
plain python3) can import it.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSON-lines file; a missing file yields an empty list.

    Iterates file lines (LF only): question/response text may contain
    Unicode separators (e.g. U+2028) that splitlines() would wrongly
    treat as record boundaries.
    """
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def safe_tag(text: str) -> str:
    """Filesystem-safe tag: letters/digits/dots/dashes/underscores."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def model_tag(model: dict | None, compat: dict) -> str:
    repo = (model or {}).get("repo") or compat.get("model_repo") or "model"
    rev = (model or {}).get("revision") or compat.get("model_revision") or ""
    return safe_tag(f"{repo.split('/')[-1]}-{rev[:8]}" if rev else repo)
