#!/usr/bin/env python3
"""Inspect the local inference setup for an mlx_local model config --
WITHOUT loading the model or running inference.

Reports:
  * hardware: chip, total RAM, free disk;
  * whether the pinned snapshot is already in the HuggingFace cache
    (reads cache directory names only; no network, no download);
  * a rough fit judgment (4-bit ~8B weights ~= 4-5 GB) against RAM;
  * whether the generation venv (.venv-mlx) exists.

Usage:
    python3 scripts/check_local_setup.py \
        --model-config configs/model_llama-3.1-8b-instruct-4bit.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import backends

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Rough on-disk / in-RAM size class for 4-bit MLX 7-8B instruct models,
# for reporting only (not a hard limit).
APPROX_WEIGHT_BYTES = 5 * 1024**3


def mem_total_bytes() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except Exception:
        return None


def snapshot_cached(repo_id: str, revision: str | None) -> dict:
    cache_name = "models--" + repo_id.replace("/", "--")
    snaps = HF_CACHE / cache_name / "snapshots"
    found = []
    if snaps.is_dir():
        found = [p.name for p in snaps.iterdir() if p.is_dir()]
    return {
        "cache_dir": str(snaps),
        "cached_revisions": found,
        "pinned_present": (revision in found) if revision else bool(found),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", type=Path, required=True)
    args = ap.parse_args()

    cfg = backends.load_model_config(args.model_config)
    if cfg.get("backend", "mlx_local") != "mlx_local":
        print(f"config backend is {cfg.get('backend')!r}; this check is "
              "for mlx_local models only.", file=sys.stderr)
        return 2

    repo, rev = cfg["repo_id"], cfg.get("revision")
    print(f"model: {repo}@{rev}")

    mem = mem_total_bytes()
    if mem:
        print(f"hardware: {platform_model()}, "
              f"{mem / 1024**3:.0f} GB unified memory")
        headroom = ("comfortable" if mem >= 16 * 1024**3
                    else "tight (close other apps)")
        print(f"fit estimate: ~{APPROX_WEIGHT_BYTES // 1024**3} GB 4-bit "
              f"weights + KV cache vs {mem // 1024**3} GB RAM -- {headroom}"
              " (no inference performed)")
    du = shutil.disk_usage(Path.home())
    print(f"disk free: {du.free / 1024**3:.0f} GB")

    snap = snapshot_cached(repo, rev)
    if snap["pinned_present"]:
        print(f"weights: pinned snapshot already cached under {snap['cache_dir']}")
    else:
        print(f"weights: NOT cached ({snap['cache_dir']} has "
              f"{snap['cached_revisions'] or 'no snapshots'})")
        print(f"manual download (~5 GB), inside .venv-mlx:")
        print(f"  .venv-mlx/bin/python -c \"import huggingface_hub; "
              f"print(huggingface_hub.snapshot_download('{repo}', "
              f"revision='{rev}'))\"")
        print("  (Llama weights are under the Llama 3.1 Community "
              "License; the mlx-community repo is not gated, but the "
              "license still applies to your use.)")

    venv = REPO_ROOT / ".venv-mlx" / "bin" / "python"
    print(f"generation venv (.venv-mlx): "
          f"{'present' if venv.exists() else 'MISSING -- create per README'}")
    return 0 if snap["pinned_present"] else 1


def platform_model() -> str:
    for key in ("hw.model", "machdep.cpu.brand_string"):
        try:
            out = subprocess.run(["sysctl", "-n", key],
                                 capture_output=True, text=True,
                                 check=True)
            if out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            continue
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
