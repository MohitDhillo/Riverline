"""Agent 1 — Assessment (chat).

Goal: verify identity, capture debt acknowledgement, capture financial situation.
Outcome states: 'assessed' | 'partial' | 'no_response'.

Outcome classification on Day 1 is a lightweight heuristic on transcript content.
On Day 2 we move to tool-call-based outcomes (verify_identity + record_disclosure
return structured records that drive classification deterministically).
"""

from __future__ import annotations

import re

from packages.agents.base import BaseAgent


class AssessmentAgent(BaseAgent):
    agent_id = "agent_1"
    max_tokens_out = 320  # short turns

    def classify_outcome(self, transcript: list[dict]) -> str:
        if not transcript:
            return "no_response"

        borrower_turns = [t for t in transcript if t["role"] == "user"]
        agent_turns = [t for t in transcript if t["role"] == "assistant"]
        if not borrower_turns:
            return "no_response"

        joined_borrower = " ".join(t["content"].lower() for t in borrower_turns)
        joined_agent = " ".join(t["content"].lower() for t in agent_turns)

        # crude signals — replaced by tool calls in Day 2
        identity_signal = (
            re.search(r"\b\d{4}\b", joined_borrower) is not None
            and ("dob" in joined_borrower or re.search(r"\b(19|20)\d{2}\b", joined_borrower))
        )
        income_signal = any(w in joined_borrower for w in [
            "income", "salary", "wage", "month", "$", "earn", "make", "paid", "1k", "2k", "3k", "4k", "5k",
        ])
        employment_signal = any(w in joined_borrower for w in [
            "job", "work", "employ", "unemploy", "self-employ", "part-time", "full-time", "freelance",
        ])
        captured = sum([bool(identity_signal), bool(income_signal), bool(employment_signal)])

        if captured >= 2 and len(agent_turns) >= 2:
            return "assessed"
        if captured >= 1:
            return "partial"
        return "partial" if len(borrower_turns) >= 2 else "no_response"
