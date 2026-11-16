#!/usr/bin/env python3
"""Check whether this account can see a model -- WITHOUT generating text.

Calls client.models.retrieve(<model>), which returns the model object
metadata only. It generates no tokens and is not billed.

Limitations of this check:
  * It proves the model ID resolves for this API key's project. It does
    NOT prove chat-completions permission, billing state, or that a real
    request will succeed -- the first real generation could still fail
    (e.g. quota). It is the cheapest available pre-flight check.
  * It says nothing about the account's remaining credit balance; the
    budget ledger tracks only this pipeline's calculated spend.

Reads OPENAI_API_KEY from the environment; the key is never printed.

Usage:
    python3 scripts/check_openai_access.py \
        --model-config configs/model_gpt-4o-2024-08-06.json
    python3 scripts/check_openai_access.py --model gpt-4o-2024-08-06
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import backends


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-config", type=Path, default=None)
    args = ap.parse_args()

    if args.model_config:
        cfg = backends.load_model_config(args.model_config)
        model = cfg["model"]
    elif args.model:
        model = args.model
    else:
        ap.error("give --model or --model-config")

    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is not set. Export it first "
              "(it is never printed).", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("error: the 'openai' package is not installed in this "
              "environment. Install requirements-generation.txt into "
              ".venv-mlx first.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    max_retries=0, timeout=30.0)
    try:
        m = client.models.retrieve(model)
    except Exception as e:
        print(f"NOT ACCESSIBLE: models.retrieve({model!r}) failed: "
              f"{type(e).__name__}: {e}")
        print("This does not prove the model is unusable by another "
              "route; it means this key/project cannot see this model "
              "ID. Check the model name, project, and key.")
        return 1

    print(f"ACCESSIBLE: model object retrieved: id={m.id!r} "
          f"owned_by={getattr(m, 'owned_by', None)!r}")
    print("Note: this only confirms the model ID resolves for this key. "
          "It does not verify billing, quota, or that a chat completion "
          "will succeed -- and it does not query the credit balance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
