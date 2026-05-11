"""Tool schemas + handlers for the three agents.

Tools are mostly side-effect-free *records*: the model calls `verify_identity` or
`record_commitment` and we ack. The recorded calls drive outcome classification
deterministically (replacing Day-1's regex heuristics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# -- Agent 1 (Assessment) --------------------------------------------------

VERIFY_IDENTITY = {
    "name": "verify_identity",
    "description": "Confirm the borrower's identity using partial identifiers. Call this once you have collected last-4 SSN and date of birth.",
    "input_schema": {
        "type": "object",
        "properties": {
            "last4_ssn": {"type": "string", "description": "Last 4 digits of borrower SSN"},
            "dob": {"type": "string", "description": "Date of birth as YYYY-MM-DD or MM/DD/YYYY"},
        },
        "required": ["last4_ssn", "dob"],
    },
}

RECORD_DISCLOSURE = {
    "name": "record_disclosure",
    "description": "Record one piece of borrower-disclosed financial information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "field": {
                "type": "string",
                "enum": [
                    "employment", "monthly_income_band", "ability_to_pay_lump",
                    "ability_to_pay_plan", "stated_hardship", "debt_acknowledged",
                ],
            },
            "value": {"type": "string"},
        },
        "required": ["field", "value"],
    },
}


# -- Agent 2 (Resolution) --------------------------------------------------

PRESENT_OFFER = {
    "name": "present_offer",
    "description": "Record that you presented one of the policy-approved offers to the borrower.",
    "input_schema": {
        "type": "object",
        "properties": {
            "offer_type": {
                "type": "string",
                "enum": ["lump_30", "plan_12", "hardship_referral"],
            },
            "terms": {"type": "string", "description": "One-sentence summary of the terms presented"},
        },
        "required": ["offer_type"],
    },
}

RECORD_COMMITMENT = {
    "name": "record_commitment",
    "description": "Record borrower commitment to an offer (deal_agreed outcome).",
    "input_schema": {
        "type": "object",
        "properties": {
            "offer_type": {"type": "string", "enum": ["lump_30", "plan_12", "hardship_referral"]},
            "amount": {"type": "number"},
            "due_date": {"type": "string"},
        },
        "required": ["offer_type"],
    },
}


# -- Agent 3 (Final Notice) ------------------------------------------------

ISSUE_FINAL_OFFER = {
    "name": "issue_final_offer",
    "description": "Issue the documented final offer with a hard expiry. Call this exactly once per conversation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "expiry": {"type": "string", "description": "ISO date string"},
            "accepted_by_borrower": {"type": "boolean"},
        },
        "required": ["amount", "expiry"],
    },
}

FLAG_FOR_LEGAL = {
    "name": "flag_for_legal",
    "description": "Flag the account for legal review when no resolution was reached.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}

FLAG_FOR_WRITEOFF = {
    "name": "flag_for_writeoff",
    "description": "Flag the account for write-off when collection is no longer viable.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}


# -- Shared --------------------------------------------------------------

FLAG_OPT_OUT = {
    "name": "flag_opt_out",
    "description": "Flag the borrower as opt-out (they asked to stop being contacted). Required by compliance rule 3.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "additionalProperties": False,
    },
}


# -- per-agent tool sets ---------------------------------------------------

AGENT_1_TOOLS = [VERIFY_IDENTITY, RECORD_DISCLOSURE, FLAG_OPT_OUT]
AGENT_2_TOOLS = [PRESENT_OFFER, RECORD_COMMITMENT, FLAG_OPT_OUT]
AGENT_3_TOOLS = [ISSUE_FINAL_OFFER, FLAG_FOR_LEGAL, FLAG_FOR_WRITEOFF, FLAG_OPT_OUT]


# -- recorder ---------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class ToolRecorder:
    calls: list[ToolCall] = field(default_factory=list)

    def handle(self, name: str, input_: dict) -> dict[str, Any]:
        self.calls.append(ToolCall(name=name, input=input_))
        # Simple acks — the model just needs *some* result block to keep going.
        if name == "verify_identity":
            return {"ok": True, "verified": True}
        if name == "record_disclosure":
            return {"ok": True, "recorded": input_.get("field")}
        if name == "present_offer":
            return {"ok": True, "presented": input_.get("offer_type")}
        if name == "record_commitment":
            return {"ok": True, "deal": "agreed"}
        if name == "issue_final_offer":
            return {"ok": True, "issued": True}
        if name == "flag_for_legal":
            return {"ok": True, "flagged": "legal"}
        if name == "flag_for_writeoff":
            return {"ok": True, "flagged": "writeoff"}
        if name == "flag_opt_out":
            return {"ok": True, "opt_out": True}
        return {"ok": True}

    def names(self) -> set[str]:
        return {c.name for c in self.calls}

    def by_name(self, name: str) -> list[ToolCall]:
        return [c for c in self.calls if c.name == name]
