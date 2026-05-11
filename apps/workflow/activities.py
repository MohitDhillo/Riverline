"""Temporal activities.

Activities are the *only* place LLM/network/db calls live. Workflows must be
deterministic — they cannot call LLMs, read the clock, or generate randomness
outside `workflow.now()` / `workflow.random()`.

Day 1: just the chat-agent activity. Day 2 adds the voice activity and summarizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from temporalio import activity

from packages.agents.agent_1 import AssessmentAgent
from packages.simulator.borrower import BorrowerSimulator, load_borrowers
from packages.simulator.runner import run_chat_conversation
from packages.storage.repos import install_cost_persistence


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


_AGENT_CLASSES = {
    "agent_1": AssessmentAgent,
    # agent_3 added Day 2
}


@activity.defn
async def run_chat_agent(inp: ChatAgentInput) -> ChatAgentOutput:
    """Drive a full chat between the named agent and a simulated borrower.

    Day 1 uses the simulator as the borrower for end-to-end testing. The
    production path (Day 2+) will instead consume real user messages via
    Temporal signals.
    """
    install_cost_persistence()
    activity.heartbeat()

    agent_cls = _AGENT_CLASSES.get(inp.agent_id)
    if agent_cls is None:
        raise RuntimeError(f"unknown agent_id: {inp.agent_id}")

    # Pick the borrower from the seeded fixtures.
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
