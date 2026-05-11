"""Trim-logic tests for the handoff payload.

These run without an LLM — they verify the deterministic trim order and
budget enforcement on a constructed payload.
"""

from __future__ import annotations

import pytest

from packages.llm.token_guard import HANDOFF_BUDGET, count_tokens
from packages.summarizer.schema import (
    ComplianceFlags,
    Debt,
    FinancialSituation,
    HandoffPayload,
    Identity,
    OfferMade,
)
from packages.summarizer.trim import trim_to_budget


_VARIED_LINE = (
    "The borrower expressed frustration about the proposed timeline and noted that "
    "their household budget cannot absorb another fixed obligation right now without "
    "first reviewing existing childcare and prescription medication expenses fully."
)


def _make_big_payload() -> HandoffPayload:
    return HandoffPayload(
        identity=Identity(verified=True, method="last4_ssn+dob", confidence="high"),
        debt=Debt(amount_acknowledged=4250.00, borrower_disputes=False, dispute_basis=None),
        financial_situation=FinancialSituation(
            employment="part_time",
            monthly_income_band="1k-2k",
            stated_hardship=["medical_bills", "reduced_hours"],
            ability_to_pay_lump="no",
            ability_to_pay_plan="yes_under_200_mo",
        ),
        offers_made=[
            OfferMade(type="lump_30", borrower_response="declined"),
            OfferMade(type="plan_12", borrower_response="considering"),
        ],
        objections_raised=[f"objection_{i}: {_VARIED_LINE}" for i in range(6)],
        emotional_state="frustrated_but_engaged",
        compliance_flags=ComplianceFlags(
            opt_out_requested=False,
            hardship_program_offered=True,
            sensitive_disclosure="medical",
        ),
        open_threads=[f"thread_{i}: {_VARIED_LINE}" for i in range(3)],
        borrower_quotes=[
            "I literally cannot do three hundred a month right now please listen",
            "I do want this resolved you have my word on that I promise",
            "I just need a couple weeks to figure things out alright thanks",
        ],
    )


def test_compact_payload_already_within_budget() -> None:
    payload = HandoffPayload(
        identity=Identity(verified=True, method="last4+dob", confidence="high"),
        emotional_state="cooperative",
    )
    out = trim_to_budget(payload)
    assert out.payload_tokens <= HANDOFF_BUDGET
    assert out.trimmed_fields == {}


def test_trim_drops_quotes_first() -> None:
    payload = _make_big_payload()
    before = count_tokens(payload.to_compact_json())
    assert before > HANDOFF_BUDGET  # sanity: our fixture is actually too large
    out = trim_to_budget(payload)
    assert out.payload_tokens <= HANDOFF_BUDGET
    # borrower_quotes is the largest trimmable field, dropped first
    assert "borrower_quotes" in out.trimmed_fields
    # original payload not mutated
    assert len(payload.borrower_quotes) == 3


def test_trim_preserves_identity_and_compliance_flags() -> None:
    payload = _make_big_payload()
    out = trim_to_budget(payload)
    assert out.payload.identity.verified is True
    assert out.payload.compliance_flags.opt_out_requested is False
    assert out.payload.compliance_flags.hardship_program_offered is True
    assert out.payload.compliance_flags.sensitive_disclosure == "medical"


def test_trim_raises_when_locked_fields_exceed_budget() -> None:
    payload = HandoffPayload(
        identity=Identity(verified=True, method="x " * 200, confidence="high"),
        emotional_state="x " * 200,
    )
    with pytest.raises(ValueError, match="exhausting all trimmable"):
        trim_to_budget(payload, budget=50)


def test_trim_order_respected_when_quotes_alone_insufficient() -> None:
    # build a payload where dropping just quotes is not enough — objections must go too
    payload = HandoffPayload(
        identity=Identity(verified=True, method="last4+dob", confidence="high"),
        objections_raised=[f"objection_{i}: {_VARIED_LINE}" for i in range(10)],
        borrower_quotes=[
            "quote one with actual borrower words yes",
            "quote two with actual borrower words too",
            "quote three with actual borrower words also",
        ],
        open_threads=["thread A about the spouse decision", "thread B about employer paperwork"],
    )
    assert count_tokens(payload.to_compact_json()) > HANDOFF_BUDGET  # sanity
    out = trim_to_budget(payload)
    assert out.payload_tokens <= HANDOFF_BUDGET
    # quotes are dropped first
    assert out.trimmed_fields.get("borrower_quotes", 0) >= 1
    # objections dropped second
    assert out.trimmed_fields.get("objections_raised", 0) >= 1
