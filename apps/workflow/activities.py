"""Temporal activities for the collections pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from temporalio import activity

from packages.agents.agent_1 import AssessmentAgent
from packages.agents.agent_2 import ResolutionAgent
from packages.agents.agent_3 import FinalNoticeAgent
from packages.simulator.borrower import BorrowerSimulator, load_borrowers
from packages.simulator.runner import run_chat_conversation
from packages.storage.repos import (
    install_cost_persistence,
    load_turns,
    record_handoff,
)
from packages.summarizer import summarize_for_handoff


@dataclass
class ChatAgentInput:
    borrower_id: str
    agent_id: str
    handoff: str = ""
    iteration_id: Optional[int] = None
    max_turns: int = 12


@dataclass
class ChatAgentOutput:
    conversation_id: str
    outcome: str
    turns: int
    transcript: list[dict]


@dataclass
class SummarizeInput:
    conversation_ids: list[str]   # in order; second one is the latest stage
    to_agent: str                  # 'to_agent_2' | 'to_agent_3'
    iteration_id: Optional[int] = None


@dataclass
class SummarizeOutput:
    payload_json: str
    payload_tokens: int
    trimmed_fields: dict


_AGENT_CLASSES = {
    "agent_1": AssessmentAgent,
    "agent_2": ResolutionAgent,
    "agent_3": FinalNoticeAgent,
}


@activity.defn
async def run_chat_agent(inp: ChatAgentInput) -> ChatAgentOutput:
    install_cost_persistence()
    activity.heartbeat()

    agent_cls = _AGENT_CLASSES.get(inp.agent_id)
    if agent_cls is None:
        raise RuntimeError(f"unknown agent_id: {inp.agent_id}")

    candidates = [b for b in load_borrowers() if b.id == inp.borrower_id]
    if not candidates:
        raise RuntimeError(f"borrower {inp.borrower_id} not found in seeds")
    profile = candidates[0]

    agent = agent_cls()
    sim = BorrowerSimulator(profile)
    conv_id, result = run_chat_conversation(
        agent,
        sim,
        max_turns=inp.max_turns,
        handoff=inp.handoff,
        workflow_id=activity.info().workflow_id,
        iteration_id=inp.iteration_id,
    )
    return ChatAgentOutput(
        conversation_id=str(conv_id),
        outcome=result.outcome,
        turns=result.turns,
        transcript=result.transcript,
    )


@activity.defn
async def summarize_handoff(inp: SummarizeInput) -> SummarizeOutput:
    install_cost_persistence()
    activity.heartbeat()

    # combine turns across all listed conversations, in order
    combined: list[dict] = []
    for cid in inp.conversation_ids:
        combined.extend(load_turns(uuid.UUID(cid)))

    target_conv = uuid.UUID(inp.conversation_ids[-1])
    res = summarize_for_handoff(
        combined,
        to_agent=inp.to_agent,
        conversation_id=str(target_conv),
        iteration_id=inp.iteration_id,
    )

    payload_json = res.payload.to_compact_json()
    record_handoff(
        conversation_id=target_conv,
        from_agent=combined[-1].get("agent_id", "unknown") if combined else "unknown",
        to_agent=inp.to_agent,
        payload=res.payload.model_dump(),
        payload_tokens=res.payload_tokens,
        trimmed_fields=res.trimmed_fields or None,
    )
    return SummarizeOutput(
        payload_json=payload_json,
        payload_tokens=res.payload_tokens,
        trimmed_fields=res.trimmed_fields,
    )
