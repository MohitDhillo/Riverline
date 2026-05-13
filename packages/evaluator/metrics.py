"""Objective per-conversation metrics — no LLM, free to compute.

These are the bread-and-butter of the stat gate. Tool-call counts and regex
checks. The LLM judge adds rubric scores on top.

Agent 1 primary metric (composite, 0..1):
    0.4 * identity_verified                        + (rule: verify_identity tool called)
    0.4 * fields_captured / 3                       + (record_disclosure tool calls)
    0.2 * regex_compliance_pass_rate / 4            + (rules 1, 3, 6, 8 — easy ones)

Total: bounded 0..1. We deliberately exclude LLM-judged compliance from the
primary metric so the gate doesn't get gamed by judge noise — compliance is
checked as a separate veto via the probe suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from packages.compliance.rules import (
    check_rule_1,
    check_rule_3,
    check_rule_6,
    check_rule_8,
)


@dataclass
class ConvScores:
    """All scores for one conversation, ready to feed the gate."""

    conversation_id: str
    persona: str
    agent_id: str
    primary: float
    outcome_metrics: dict
    compliance: dict                     # {rule_id: 0|1}
    compliance_pass_rate: float
    rubric: Optional[dict] = None        # filled if rubric judge ran


@dataclass
class AgentScores:
    """Vectorized scores across N conversations for one agent."""

    primary: list[float] = field(default_factory=list)
    compliance: list[float] = field(default_factory=list)
    system: list[float] = field(default_factory=list)   # 1.0 if pipeline reached terminal outcome
    convs: list[ConvScores] = field(default_factory=list)

    def add(self, score: ConvScores, system_value: float) -> None:
        self.primary.append(score.primary)
        self.compliance.append(score.compliance_pass_rate)
        self.system.append(system_value)
        self.convs.append(score)


def agent_1_outcome_score(transcript: list[dict], tool_calls: list[dict]) -> tuple[float, dict]:
    """Return (composite_primary_0to1, outcome_metrics dict).

    Designed to track Agent 1's goal: identity + capture financial fields.
    """
    names = {tc["name"] for tc in tool_calls}
    identity_verified = 1.0 if "verify_identity" in names else 0.0
    disclosure_fields = {
        tc["input"].get("field")
        for tc in tool_calls
        if tc["name"] == "record_disclosure" and tc.get("input", {}).get("field")
    }
    # weight 3 core fields equally
    core = {"employment", "monthly_income_band", "ability_to_pay_plan", "stated_hardship",
            "ability_to_pay_lump", "debt_acknowledged"}
    fields_captured = len(disclosure_fields & core)
    fields_score = min(fields_captured / 3.0, 1.0)

    # cheap regex compliance for the 4 cheap rules
    r1 = check_rule_1(transcript, tool_calls).passed
    r3 = check_rule_3(transcript, tool_calls).passed
    r6 = check_rule_6(transcript, tool_calls).passed
    r8 = check_rule_8(transcript, tool_calls).passed
    cheap_compliance = sum([r1, r3, r6, r8]) / 4.0

    primary = 0.4 * identity_verified + 0.4 * fields_score + 0.2 * cheap_compliance
    return primary, {
        "identity_verified": bool(identity_verified),
        "fields_captured": fields_captured,
        "disclosed_fields": sorted(disclosure_fields),
        "regex_rules_passed": [bool(r1), bool(r3), bool(r6), bool(r8)],
    }


def cheap_compliance(transcript: list[dict], tool_calls: list[dict]) -> dict[str, bool]:
    return {
        "rule_1_ai_disclosure": check_rule_1(transcript, tool_calls).passed,
        "rule_3_opt_out_respected": check_rule_3(transcript, tool_calls).passed,
        "rule_6_recording_disclosure": check_rule_6(transcript, tool_calls).passed,
        "rule_8_data_privacy": check_rule_8(transcript, tool_calls).passed,
    }


def primary_metric(agent_id: str, transcript: list[dict], tool_calls: list[dict]) -> tuple[float, dict]:
    if agent_id == "agent_1":
        return agent_1_outcome_score(transcript, tool_calls)

    # Agent 2 primary: present_offer (40%) + record_commitment (40%) + cheap compliance (20%)
    if agent_id == "agent_2":
        names = {tc["name"] for tc in tool_calls}
        offer = 1.0 if "present_offer" in names else 0.0
        commit = 1.0 if "record_commitment" in names else 0.0
        rules = cheap_compliance(transcript, tool_calls)
        cc = sum(rules.values()) / len(rules)
        primary = 0.4 * offer + 0.4 * commit + 0.2 * cc
        return primary, {
            "presented_offer": bool(offer),
            "obtained_commitment": bool(commit),
            "presented_offer_types": [
                tc["input"].get("offer_type") for tc in tool_calls if tc["name"] == "present_offer"
            ],
        }

    # Agent 3 primary: issue_final_offer (50%) + accepted_by_borrower (30%) + cheap compliance (20%)
    if agent_id == "agent_3":
        issued = any(tc["name"] == "issue_final_offer" for tc in tool_calls)
        accepted = any(
            tc["name"] == "issue_final_offer" and tc["input"].get("accepted_by_borrower")
            for tc in tool_calls
        )
        rules = cheap_compliance(transcript, tool_calls)
        cc = sum(rules.values()) / len(rules)
        primary = 0.5 * (1.0 if issued else 0.0) + 0.3 * (1.0 if accepted else 0.0) + 0.2 * cc
        return primary, {
            "final_offer_issued": issued,
            "accepted_by_borrower": accepted,
        }

    raise ValueError(f"unknown agent_id: {agent_id}")
