"""Deterministic 500-token trim for a HandoffPayload.

Trim order (per FINAL_PLAN §4):
    1) drop borrower_quotes one at a time (oldest last — keep the most recent)
    2) drop objections_raised one at a time (oldest first)
    3) drop open_threads one at a time (oldest first)
    4) if still over → raise (this means the structured non-trimmable fields are too large)

We never trim: identity, debt, financial_situation, offers_made,
emotional_state, compliance_flags.
"""

from __future__ import annotations

from dataclasses import dataclass

from packages.llm.token_guard import HANDOFF_BUDGET, count_tokens
from packages.summarizer.schema import HandoffPayload


@dataclass
class TrimResult:
    payload: HandoffPayload
    payload_tokens: int
    trimmed_fields: dict


def _tokens(p: HandoffPayload) -> int:
    return count_tokens(p.to_compact_json())


def trim_to_budget(payload: HandoffPayload, budget: int = HANDOFF_BUDGET) -> TrimResult:
    p = payload.model_copy(deep=True)
    trimmed: dict[str, int] = {}

    while _tokens(p) > budget and p.borrower_quotes:
        p.borrower_quotes.pop()  # drop the most-recently-listed quote first
        trimmed["borrower_quotes"] = trimmed.get("borrower_quotes", 0) + 1

    while _tokens(p) > budget and p.objections_raised:
        p.objections_raised.pop(0)  # drop oldest objection
        trimmed["objections_raised"] = trimmed.get("objections_raised", 0) + 1

    while _tokens(p) > budget and p.open_threads:
        p.open_threads.pop(0)
        trimmed["open_threads"] = trimmed.get("open_threads", 0) + 1

    tok = _tokens(p)
    if tok > budget:
        raise ValueError(
            f"handoff is {tok} tokens after exhausting all trimmable fields "
            f"(budget {budget}); locked fields are too large"
        )

    return TrimResult(payload=p, payload_tokens=tok, trimmed_fields=trimmed)
