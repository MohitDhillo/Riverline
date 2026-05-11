"""Cost ledger + hard kill switch.

Every LLM call must go through ``BudgetTracker.record(...)``. When the running total
crosses ``BUDGET_KILL_USD`` (default $18 of $20), ``BudgetExhausted`` is raised and
the caller is expected to checkpoint and exit cleanly.

Anthropic pricing (per million tokens; refreshed May 2026 — verify before submission):
    Haiku 4.5:  input $1.00 / output $5.00 / cache write $1.25 / cache read $0.10
    Sonnet 4.6: input $3.00 / output $15.00 / cache write $3.75 / cache read $0.30
    Opus 4.7:   input $15.00 / output $75.00 / cache write $18.75 / cache read $1.50
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-4-7": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
}


class BudgetExhausted(Exception):
    """Raised when the cumulative spend has crossed the kill threshold."""


@dataclass
class CostRecord:
    timestamp: datetime
    provider: str
    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    cost_usd: float
    iteration_id: Optional[int] = None
    conversation_id: Optional[str] = None


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    p = PRICING.get(model)
    if p is None:
        # unknown model — fall back to Sonnet pricing to be conservative
        p = PRICING["claude-sonnet-4-6"]
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_write_tokens * p["cache_write"]
        + cache_read_tokens * p["cache_read"]
    ) / 1_000_000.0


class BudgetTracker:
    """Process-local ledger. On Day 1 keeps records in memory; later persists to Postgres."""

    def __init__(self, total_usd: float = 20.0, kill_usd: float = 18.0) -> None:
        self.total_usd = total_usd
        self.kill_usd = kill_usd
        self._lock = threading.Lock()
        self._spent = 0.0
        self._records: list[CostRecord] = []
        # Hook installed later by storage layer to mirror records to Postgres
        self._persist_hook = None

    def install_persist_hook(self, fn) -> None:
        self._persist_hook = fn

    def record(
        self,
        provider: str,
        model: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        iteration_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> CostRecord:
        cost = compute_cost(
            model, input_tokens, output_tokens, cache_write_tokens, cache_read_tokens
        )
        rec = CostRecord(
            timestamp=datetime.utcnow(),
            provider=provider,
            model=model,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost,
            iteration_id=iteration_id,
            conversation_id=conversation_id,
        )
        with self._lock:
            self._spent += cost
            self._records.append(rec)
            crossed = self._spent >= self.kill_usd
        if self._persist_hook:
            try:
                self._persist_hook(rec)
            except Exception:
                pass  # never let persistence kill the call
        if crossed:
            raise BudgetExhausted(
                f"spend ${self._spent:.4f} crossed kill threshold ${self.kill_usd:.2f}"
            )
        return rec

    def spent(self) -> float:
        with self._lock:
            return self._spent

    def records(self) -> list[CostRecord]:
        with self._lock:
            return list(self._records)

    def by_purpose(self) -> dict[str, float]:
        out: dict[str, float] = {}
        with self._lock:
            for r in self._records:
                out[r.purpose] = out.get(r.purpose, 0.0) + r.cost_usd
        return out


# Process-singleton — every LLM call hits this.
_singleton: Optional[BudgetTracker] = None
_singleton_lock = threading.Lock()


def budget() -> BudgetTracker:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                from packages.config import settings

                s = settings()
                _singleton = BudgetTracker(total_usd=s.budget_total_usd, kill_usd=s.budget_kill_usd)
    return _singleton
