"""Locked handoff JSON schema — FINAL_PLAN §4.

Trim-order when over 500 tokens:
    borrower_quotes  -> objections_raised (oldest first) -> open_threads
Never trim: identity, debt, compliance_flags, emotional_state, offers_made.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Identity(BaseModel):
    verified: bool = False
    method: str = ""              # e.g., "last4_ssn + dob"
    confidence: Literal["low", "medium", "high"] = "low"


class Debt(BaseModel):
    amount_acknowledged: Optional[float] = None
    borrower_disputes: bool = False
    dispute_basis: Optional[str] = None


class FinancialSituation(BaseModel):
    employment: Optional[str] = None         # full_time | part_time | self_employed | unemployed | unknown
    monthly_income_band: Optional[str] = None # "1k-2k" etc.
    stated_hardship: list[str] = Field(default_factory=list)  # medical, job_loss, family, none, ...
    ability_to_pay_lump: Optional[str] = None    # yes | no | unknown
    ability_to_pay_plan: Optional[str] = None    # e.g., "yes_under_200_mo" | "no" | "unknown"


class OfferMade(BaseModel):
    type: str                             # lump_30 | plan_12 | hardship_referral | final_offer
    borrower_response: str                 # declined | considering | accepted | counter


class ComplianceFlags(BaseModel):
    opt_out_requested: bool = False
    hardship_program_offered: bool = False
    sensitive_disclosure: Optional[str] = None   # medical | job_loss | family_emergency | none


class HandoffPayload(BaseModel):
    identity: Identity = Field(default_factory=Identity)
    debt: Debt = Field(default_factory=Debt)
    financial_situation: FinancialSituation = Field(default_factory=FinancialSituation)
    offers_made: list[OfferMade] = Field(default_factory=list)
    objections_raised: list[str] = Field(default_factory=list)
    emotional_state: str = "neutral"
    compliance_flags: ComplianceFlags = Field(default_factory=ComplianceFlags)
    open_threads: list[str] = Field(default_factory=list)
    borrower_quotes: list[str] = Field(default_factory=list, max_length=3)

    def to_compact_json(self) -> str:
        """Smallest valid JSON serialization (used for token counting + storage)."""
        return self.model_dump_json(exclude_none=False)
