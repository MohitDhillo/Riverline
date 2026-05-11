"""Compliance probe runner.

Loads `data/compliance_probes.json`, runs each probe against the named agent
with a scripted borrower, evaluates the targeted rule, and aggregates results.

This is the **pre-flight gate** in the self-learning loop: a candidate prompt
must score 100% on every rule before it's eligible for paired evaluation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packages.agents.base import BaseAgent
from packages.agents.agent_1 import AssessmentAgent
from packages.agents.agent_2 import ResolutionAgent
from packages.agents.agent_3 import FinalNoticeAgent
from packages.compliance.rules import RuleResult, check_rule
from packages.compliance.scripted_borrower import ScriptedBorrower
from packages.llm import AnthropicClient
from packages.storage.repos import add_turn, create_conversation, end_conversation

PROBES_FILE = Path(__file__).resolve().parents[2] / "data" / "compliance_probes.json"

_AGENT_CLASSES = {
    "agent_1": AssessmentAgent,
    "agent_2": ResolutionAgent,
    "agent_3": FinalNoticeAgent,
}


@dataclass
class ProbeOutcome:
    probe_id: str
    rule_id: str
    agent_id: str
    passed: bool
    transcript: list[dict]
    tool_calls: list[dict]
    evidence: dict


@dataclass
class ProbeSuiteResult:
    outcomes: list[ProbeOutcome] = field(default_factory=list)

    def all_pass(self) -> bool:
        return all(o.passed for o in self.outcomes)

    def by_rule(self) -> dict[str, list[ProbeOutcome]]:
        out: dict[str, list[ProbeOutcome]] = {}
        for o in self.outcomes:
            out.setdefault(o.rule_id, []).append(o)
        return out

    def failed(self) -> list[ProbeOutcome]:
        return [o for o in self.outcomes if not o.passed]

    def summary(self) -> str:
        lines = []
        for rule, items in sorted(self.by_rule().items()):
            ok = sum(1 for i in items if i.passed)
            lines.append(f"  {rule:35s} {ok}/{len(items)}")
        return "\n".join(lines)


def load_probes() -> list[dict]:
    return json.loads(PROBES_FILE.read_text())["probes"]


def _run_one_agent_probe(
    probe: dict,
    *,
    judge_client: Optional[AnthropicClient] = None,
) -> ProbeOutcome:
    agent_id = probe["agent_id"]
    agent_cls = _AGENT_CLASSES[agent_id]
    agent: BaseAgent = agent_cls()

    borrower = ScriptedBorrower(probe["scripted_borrower_turns"], probe_id=probe["id"])
    max_turns = probe.get("max_agent_turns", 4)

    conv_id = create_conversation(
        borrower_id=uuid.UUID(borrower.profile.id),
        persona=borrower.profile.persona,
    )

    history: list[dict] = []
    seq = 0
    all_tool_calls: list[dict] = []

    for _ in range(max_turns):
        agent_reply = agent.reply(
            history=history,
            handoff="",
            conversation_id=str(conv_id),
        )
        agent_msg = agent_reply.text or "(no response)"
        history.append({"role": "assistant", "content": agent_msg})
        seq += 1
        add_turn(
            conversation_id=conv_id,
            seq=seq,
            agent_id=agent_id,
            role="assistant",
            content=agent_msg,
            token_counts=agent_reply.token_counts,
            tool_calls={"calls": agent_reply.tool_calls} if agent_reply.tool_calls else None,
        )
        all_tool_calls.extend(agent_reply.tool_calls)

        if any(tc["name"] == "flag_opt_out" for tc in agent_reply.tool_calls):
            break
        if any(p in agent_msg.lower() for p in [
            "thank you for your time", "you will be contacted", "you will receive",
            "goodbye", "have a good day",
        ]):
            break

        borrower_msg = borrower.reply(history=history)
        history.append({"role": "user", "content": borrower_msg})
        seq += 1
        add_turn(
            conversation_id=conv_id,
            seq=seq,
            agent_id="borrower",
            role="user",
            content=borrower_msg,
        )

    end_conversation(conv_id, "probe_complete")

    rule_id = probe["rule"]
    result: RuleResult = check_rule(rule_id, history, all_tool_calls, judge_client)
    return ProbeOutcome(
        probe_id=probe["id"],
        rule_id=rule_id,
        agent_id=agent_id,
        passed=result.passed,
        transcript=history,
        tool_calls=all_tool_calls,
        evidence=result.evidence,
    )


def run_probe(probe_id: str, judge_client: Optional[AnthropicClient] = None) -> ProbeOutcome:
    probes = {p["id"]: p for p in load_probes()}
    if probe_id not in probes:
        raise ValueError(f"unknown probe: {probe_id}")
    return _run_one_agent_probe(probes[probe_id], judge_client=judge_client)


def run_probe_suite(
    only_rules: Optional[list[str]] = None,
    judge_client: Optional[AnthropicClient] = None,
) -> ProbeSuiteResult:
    """Run all probes (or filter to specific rules)."""
    probes = load_probes()
    if only_rules:
        probes = [p for p in probes if p["rule"] in only_rules]
    judge_client = judge_client or AnthropicClient()
    suite = ProbeSuiteResult()
    for probe in probes:
        try:
            outcome = _run_one_agent_probe(probe, judge_client=judge_client)
        except Exception as e:  # don't let one probe kill the suite
            outcome = ProbeOutcome(
                probe_id=probe["id"],
                rule_id=probe["rule"],
                agent_id=probe["agent_id"],
                passed=False,
                transcript=[],
                tool_calls=[],
                evidence={"error": str(e)},
            )
        suite.outcomes.append(outcome)
    return suite
