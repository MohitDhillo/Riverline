"""Tool schemas + handlers for the three agents.

Tools are mostly side-effect-free *records* — the model calls `verify_identity` or
`record_commitment` and we ack. Recorded calls drive outcome classification
deterministically.

`verify_identity` is the exception: it does a REAL check against the seeded borrower
profile (passed in via ToolRecorder.verification_data). Without that data the legacy
behavior is to accept (used by compliance probes where there's no borrower profile).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# -- Agent 1 (Assessment) --------------------------------------------------

VERIFY_IDENTITY = {
    "name": "verify_identity",
    "description": (
        "Confirm the borrower's identity. Single-piece flow: pass `last4_ssn` "
        "alone first. If the tool returns verified=false, ask the borrower for "
        "DOB and re-call with both `last4_ssn` AND `dob`. Never ask for full SSN."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "last4_ssn": {"type": "string", "description": "Last 4 digits of borrower SSN"},
            "dob": {"type": "string", "description": "Date of birth (optional, escalation only). Accepts YYYY-MM-DD or MM/DD/YYYY."},
        },
        "required": ["last4_ssn"],
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
    "description": (
        "Issue the documented final offer with a hard expiry. Call this once at "
        "the start of the conversation with accepted_by_borrower=false. If the "
        "borrower later accepts, call it again with accepted_by_borrower=true."
    ),
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

END_CONVERSATION = {
    "name": "end_conversation",
    "description": (
        "Cleanly terminate the conversation when there is nothing more to do. "
        "Call this AFTER your final borrower-facing message. Examples: identity "
        "verification fully failed, task complete and you've said goodbye, or any "
        "other state where continuing would be theater. The runner stops "
        "soliciting borrower input immediately after this fires."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": [
                    "identity_unverified",
                    "task_complete",
                    "deal_agreed",
                    "no_resolution",
                    "borrower_unreachable",
                    "other",
                ],
            },
        },
        "required": ["reason"],
    },
}


# -- per-agent tool sets ---------------------------------------------------

AGENT_1_TOOLS = [VERIFY_IDENTITY, RECORD_DISCLOSURE, FLAG_OPT_OUT, END_CONVERSATION]
AGENT_2_TOOLS = [PRESENT_OFFER, RECORD_COMMITMENT, FLAG_OPT_OUT, END_CONVERSATION]
AGENT_3_TOOLS = [ISSUE_FINAL_OFFER, FLAG_FOR_LEGAL, FLAG_FOR_WRITEOFF, FLAG_OPT_OUT, END_CONVERSATION]


# -- recorder ---------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict


def _normalize_dob(raw: str) -> str:
    """Accept YYYY-MM-DD, MM/DD/YYYY, M/D/YYYY → YYYY-MM-DD."""
    if not raw:
        return ""
    raw = raw.strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", raw)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", raw)
    if m:
        mo, d, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return raw  # fallback — strict compare will likely fail and that's fine


@dataclass
class ToolRecorder:
    """Records every tool call and (for verify_identity) actually verifies.

    ``verification_data`` is the borrower's true identity record:
        {"last4_ssn": "9245", "dob": "1984-02-04", "name": "Morgan Khan"}
    Passed in from the conversation runner. When None (e.g. compliance probes
    where the scripted borrower has no real profile), verify_identity accepts
    everything — this matches the legacy behavior the probes were written against.
    """
    verification_data: Optional[dict] = None
    calls: list[ToolCall] = field(default_factory=list)

    def handle(self, name: str, input_: dict) -> dict[str, Any]:
        self.calls.append(ToolCall(name=name, input=input_))
        if name == "verify_identity":
            return self._verify(input_)
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
        if name == "end_conversation":
            return {"ok": True, "ended": True, "reason": input_.get("reason", "other")}
        return {"ok": True}

    def _verify(self, input_: dict) -> dict[str, Any]:
        if not self.verification_data:
            # No profile attached (e.g. compliance probes) — legacy accept.
            return {"ok": True, "verified": True, "method": "no_profile_attached"}

        provided_last4 = (input_.get("last4_ssn") or "").strip()
        provided_dob = (input_.get("dob") or "").strip()

        actual_last4 = str(self.verification_data.get("last4_ssn", "")).strip()
        actual_dob = _normalize_dob(str(self.verification_data.get("dob", "")))

        last4_match = bool(provided_last4) and provided_last4 == actual_last4

        # DOB is optional. If provided, must match. If not provided, accept on last4 only.
        if provided_dob:
            dob_match = _normalize_dob(provided_dob) == actual_dob
            verified = last4_match and dob_match
            reason = "" if verified else (
                "last4 mismatch" if not last4_match
                else "dob mismatch"
            )
        else:
            verified = last4_match
            reason = "" if verified else "last4 mismatch"

        return {
            "ok": True,
            "verified": verified,
            "reason": reason,
            "method": "last4+dob" if provided_dob else "last4_only",
        }

    def names(self) -> set[str]:
        return {c.name for c in self.calls}

    def by_name(self, name: str) -> list[ToolCall]:
        return [c for c in self.calls if c.name == name]
