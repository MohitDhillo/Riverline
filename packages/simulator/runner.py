"""Conversation runner — pairs a BaseAgent with a BorrowerSimulator, persists turns.

The agent speaks first. Loop until max_turns or the agent's outcome classifier says
we're done.
"""

from __future__ import annotations

import uuid
from typing import Optional

from packages.agents.base import AgentResult, BaseAgent
from packages.simulator.borrower import BorrowerSimulator
from packages.storage.repos import (
    add_turn,
    create_conversation,
    end_conversation,
)


def run_chat_conversation(
    agent: BaseAgent,
    borrower: BorrowerSimulator,
    *,
    max_turns: int = 12,
    handoff: str = "",
    workflow_id: Optional[str] = None,
    iteration_id: Optional[int] = None,
) -> tuple[uuid.UUID, AgentResult]:
    """Drive a chat between agent and borrower. Returns (conversation_id, result)."""

    conv_id = create_conversation(
        borrower_id=uuid.UUID(borrower.profile.id),
        workflow_id=workflow_id,
        persona=borrower.profile.persona,
        iteration_id=iteration_id,
        agent_versions={agent.agent_id: agent.prompt_version_num},
    )

    history: list[dict] = []
    seq = 0
    outcome = "no_response"

    # Loop: agent speaks first, then borrower, repeat.
    for turn_idx in range(max_turns):
        # ---- agent ----
        agent_reply = agent.reply(
            history=history,
            handoff=handoff,
            conversation_id=str(conv_id),
            iteration_id=iteration_id,
        )
        agent_msg = agent_reply.text or "(no response)"
        history.append({"role": "assistant", "content": agent_msg})
        seq += 1
        add_turn(
            conversation_id=conv_id,
            seq=seq,
            agent_id=agent.agent_id,
            role="assistant",
            content=agent_msg,
            token_counts=agent_reply.token_counts,
        )

        # cheap early-exit: agent signals end
        lower = agent_msg.lower()
        if any(p in lower for p in ["thank you for your time", "you will be contacted", "you will receive"]):
            # let the agent close cleanly — no further borrower response needed
            outcome = agent.classify_outcome(history)
            break

        # ---- borrower ----
        borrower_msg = borrower.reply(
            history=history,
            conversation_id=str(conv_id),
            iteration_id=iteration_id,
        ) or "(no response)"
        history.append({"role": "user", "content": borrower_msg})
        seq += 1
        add_turn(
            conversation_id=conv_id,
            seq=seq,
            agent_id="borrower",
            role="user",
            content=borrower_msg,
        )

        # check for borrower opt-out (compliance rule 3)
        if any(p in borrower_msg.lower() for p in ["stop calling", "stop contacting", "do not contact", "don't contact"]):
            # agent should handle on next turn; we just continue
            pass

    else:
        # exhausted max_turns without explicit close
        outcome = agent.classify_outcome(history)

    end_conversation(conv_id, outcome)
    return conv_id, AgentResult(outcome=outcome, turns=seq, transcript=history)
