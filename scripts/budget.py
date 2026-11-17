#!/usr/bin/env python3
"""Persistent OpenAI spending ledger and caps.

An append-only JSONL ledger (default outputs/openai_ledger.jsonl, set in
configs/budget.json) records three entry kinds per request:

  reserve     -- written BEFORE the API call, at the conservative cost
                 (estimated input tokens + full max-new-tokens output).
  commit      -- written AFTER a successful call, with the actual token
                 usage and calculated cost.
  void        -- releases a reservation (rejected request, or a human
                 releasing an unresolved one after review).
  unresolved  -- annotates a reservation whose request outcome is
                 unknown (timeout, connection drop, 5xx). The
                 reservation stays open -- and therefore counted
                 against the caps -- until reviewed and released.

Totals are computed by replay: open reservations = reserves minus
commits minus voids. A new reservation is refused when it would push
committed + open-reserved spending over the per-bucket cap
(e.g. preliminary = $5) or the project total cap ($20). A lockfile
prevents two writers racing the check-and-append; requests are
sequential, so the lock is only held for microseconds per append.

CLI:

  python3 scripts/budget.py --report
  python3 scripts/budget.py --release <reservation_id> --note "..."
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import common

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUDGET_CONFIG = REPO_ROOT / "configs" / "budget.json"


class BudgetExceeded(Exception):
    """Raised when a reservation would cross a configured cap."""


def load_budget_config(path: Path = DEFAULT_BUDGET_CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("total_cap_usd", 0.0)
    cfg.setdefault("bucket_caps_usd", {})
    cfg.setdefault("stage_to_bucket", {})
    return cfg


def _ledger_path(cfg: dict, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    p = Path(cfg.get("ledger_path", "outputs/openai_ledger.jsonl"))
    return p if p.is_absolute() else REPO_ROOT / p


@contextlib.contextmanager
def _lock(path: Path):
    """Exclusive lockfile held across a check-and-append so two
    concurrent writers cannot both pass the cap check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    fd = None
    deadline = time.time() + 30
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(
                    f"ledger lock {lock_path} held for >30s; if no run is "
                    f"active, delete the stale lock file")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        os.unlink(lock_path)


class BudgetLedger:
    def __init__(self, config_path: Path = DEFAULT_BUDGET_CONFIG,
                 ledger_path: Path | None = None):
        self.cfg = load_budget_config(config_path)
        self.path = _ledger_path(self.cfg, ledger_path)

    # -- entries -------------------------------------------------------

    def entries(self) -> list[dict]:
        return common.read_jsonl(self.path)

    def _append(self, entry: dict) -> dict:
        entry["recorded_at"] = common.utc_now()
        common.append_jsonl(self.path, entry)
        return entry

    # -- accounting ----------------------------------------------------

    def state(self) -> dict:
        """Replay the ledger: open reservations, committed spend, etc.

        Commit/void/unresolved entries inherit bucket/stage/run_id from
        their reservation so bucket caps see committed spend (older
        commits without a bucket join by reservation_id).
        """
        reserves: dict[str, dict] = {}
        open_res: dict[str, dict] = {}
        committed: list[dict] = []
        unresolved_ids: set[str] = set()
        for e in self.entries():
            t = e["type"]
            rid = e.get("reservation_id")
            if t == "reserve":
                reserves[rid] = e
                open_res[rid] = e
            elif t == "commit":
                src = reserves.get(rid, {})
                e = {**{k: src.get(k) for k in
                        ("bucket", "stage", "run_id", "model")
                        if e.get(k) is None}, **e}
                committed.append(e)
                open_res.pop(rid, None)
                unresolved_ids.discard(rid)
            elif t == "void":
                open_res.pop(rid, None)
                unresolved_ids.discard(rid)
            elif t == "unresolved":
                unresolved_ids.add(rid)
        buckets: dict[str, dict] = {}
        for e in committed:
            b = e.get("bucket") or "unbucketed"
            buckets.setdefault(b, {"committed_usd": 0.0,
                                   "open_reserved_usd": 0.0})
            buckets[b]["committed_usd"] += e.get("cost_usd", 0.0)
        for rid, e in open_res.items():
            b = e.get("bucket") or "unbucketed"
            buckets.setdefault(b, {"committed_usd": 0.0,
                                   "open_reserved_usd": 0.0})
            buckets[b]["open_reserved_usd"] += e.get("est_cost_usd", 0.0)
        return {
            "committed_usd": sum(e.get("cost_usd", 0.0) for e in committed),
            "open_reserved_usd": sum(e.get("est_cost_usd", 0.0)
                                     for e in open_res.values()),
            "open_reservations": open_res,
            "unresolved_reservation_ids": sorted(unresolved_ids
                                                 & set(open_res)),
            "commits": committed,
            "buckets": buckets,
        }

    def bucket_for_stage(self, stage: str) -> str:
        return self.cfg["stage_to_bucket"].get(stage, stage or "unbucketed")

    def would_exceed(self, stage: str, amount: float) -> str | None:
        """Return a human-readable reason if reserving `amount` would
        cross a cap, else None."""
        st = self.state()
        total_after = (st["committed_usd"] + st["open_reserved_usd"]
                       + amount)
        if total_after > self.cfg["total_cap_usd"] + 1e-9:
            return (f"total cap: committed ${st['committed_usd']:.4f} + "
                    f"open reservations ${st['open_reserved_usd']:.4f} + "
                    f"new ${amount:.4f} > "
                    f"${self.cfg['total_cap_usd']:.2f}")
        bucket = self.bucket_for_stage(stage)
        cap = self.cfg["bucket_caps_usd"].get(bucket)
        if cap is not None:
            b = st["buckets"].get(bucket, {"committed_usd": 0.0,
                                           "open_reserved_usd": 0.0})
            if (b["committed_usd"] + b["open_reserved_usd"] + amount
                    > cap + 1e-9):
                return (f"{bucket!r} bucket cap: committed "
                        f"${b['committed_usd']:.4f} + open "
                        f"${b['open_reserved_usd']:.4f} + new "
                        f"${amount:.4f} > ${cap:.2f}")
        return None

    # -- ledger operations ----------------------------------------------

    def reserve(self, *, stage: str, run_id: str, model: str,
                example_id: str, condition: str,
                est_cost_usd: float, est_input_tokens: int,
                max_output_tokens: int) -> dict:
        """Reserve conservative cost under the lock. Raises
        BudgetExceeded if the caps would be crossed."""
        rid = f"res-{uuid.uuid4().hex[:12]}"
        with _lock(self.path):
            reason = self.would_exceed(stage, est_cost_usd)
            if reason:
                raise BudgetExceeded(reason)
            return self._append({
                "type": "reserve",
                "reservation_id": rid,
                "stage": stage,
                "bucket": self.bucket_for_stage(stage),
                "run_id": run_id,
                "model": model,
                "example_id": example_id,
                "condition": condition,
                "est_cost_usd": round(est_cost_usd, 6),
                "est_input_tokens": est_input_tokens,
                "max_output_tokens": max_output_tokens,
            })

    def commit(self, reservation_id: str, *, cost_usd: float,
               usage: dict, request_id: str | None) -> dict:
        return self._append({
            "type": "commit",
            "reservation_id": reservation_id,
            "cost_usd": round(cost_usd, 6),
            "usage": usage,
            "request_id": request_id,
        })

    def mark_unresolved(self, reservation_id: str, *, run_id: str,
                        example_id: str, condition: str,
                        error: str) -> dict:
        return self._append({
            "type": "unresolved",
            "reservation_id": reservation_id,
            "run_id": run_id,
            "example_id": example_id,
            "condition": condition,
            "error": error,
        })

    def void(self, reservation_id: str, *, reason: str) -> dict:
        return self._append({
            "type": "void",
            "reservation_id": reservation_id,
            "reason": reason,
        })


# --------------------------------------------------------------------------
# Pricing helpers (model config carries the rates)
# --------------------------------------------------------------------------

def conservative_cost(model_cfg: dict, est_input_tokens: int,
                      max_output_tokens: int) -> float:
    """Worst-case reservation: estimated input + FULL output cap, at
    full price (no caching/batch discounts assumed)."""
    p = model_cfg["pricing"]
    return (est_input_tokens * p["input_usd_per_1m"] / 1e6
            + max_output_tokens * p["output_usd_per_1m"] / 1e6)


def actual_cost(model_cfg: dict, usage: dict) -> float:
    """Billed cost from reported usage."""
    p = model_cfg["pricing"]
    return ((usage.get("prompt_tokens") or 0)
            * p["input_usd_per_1m"] / 1e6
            + (usage.get("completion_tokens") or 0)
            * p["output_usd_per_1m"] / 1e6)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--release", metavar="RESERVATION_ID",
                    help="void an open reservation after manual review")
    ap.add_argument("--note", default="manual release after review")
    ap.add_argument("--config", type=Path, default=DEFAULT_BUDGET_CONFIG)
    ap.add_argument("--ledger", type=Path, default=None)
    args = ap.parse_args()

    ledger = BudgetLedger(args.config, args.ledger)

    if args.release:
        st = ledger.state()
        if args.release not in st["open_reservations"]:
            print(f"error: {args.release} is not an open reservation",
                  file=sys.stderr)
            return 1
        ledger.void(args.release, reason=args.note)
        print(f"released {args.release}")
        st = ledger.state()

    if args.report or args.release or len(sys.argv) == 1:
        st = ledger.state()
        cfg = ledger.cfg
        print(f"ledger: {ledger.path}")
        print(f"  committed spend:      ${st['committed_usd']:.4f}")
        print(f"  open reservations:    ${st['open_reserved_usd']:.4f} "
              f"({len(st['open_reservations'])} open)")
        print(f"  total cap:            ${cfg['total_cap_usd']:.2f} "
              f"(this pipeline only -- not the account balance)")
        for b, v in sorted(st["buckets"].items()):
            cap = cfg["bucket_caps_usd"].get(b)
            cap_s = f" / cap ${cap:.2f}" if cap is not None else ""
            print(f"  bucket {b!r}: committed ${v['committed_usd']:.4f}"
                  f" + open ${v['open_reserved_usd']:.4f}{cap_s}")
        if st["unresolved_reservation_ids"]:
            print(f"  UNRESOLVED (kept reserved, review then release): "
                  f"{st['unresolved_reservation_ids']}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
