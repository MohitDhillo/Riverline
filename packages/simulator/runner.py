"""Conversation runner — pairs a BaseAgent with a BorrowerSimulator, persists turns.

The agent speaks first. Loop until max_turns or the agent's outcome classifier
says we're done. Tool calls produced inside the agent are persisted alongside
the assistant turn that emitted them.
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
    conversation_id: Optional[uuid.UUID] = None,
    persona: Optional[str] = None,
) -> tuple[uuid.UUID, AgentResult]:
    """Drive a chat between agent and borrower. Returns (conversation_id, result)."""
    if conversation_id is None:
        conversation_id = create_conversation(
            borrower_id=uuid.UUID(borrower.profile.id),
            workflow_id=workflow_id,
            persona=persona or borrower.profile.persona,
            iteration_id=iteration_id,
            agent_versions={agent.agent_id: agent.prompt_version_num},
        )

    history: list[dict] = []
    seq = 0
    all_tool_calls: list[dict] = []
    outcome = "no_response"

    for _turn_idx in range(max_turns):
        # ---- agent ----
        agent_reply = agent.reply(
            history=history,
            handoff=handoff,
            conversation_id=str(conversation_id),
            iteration_id=iteration_id,
        )
        agent_msg = agent_reply.text or "(no response)"
        history.append({"role": "assistant", "content": agent_msg})
        seq += 1
        add_turn(
            conversation_id=conversation_id,
            seq=seq,
            agent_id=agent.agent_id,
            role="assistant",
            content=agent_msg,
            token_counts=agent_reply.token_counts,
            tool_calls={"calls": agent_reply.tool_calls} if agent_reply.tool_calls else None,
        )
        all_tool_calls.extend(agent_reply.tool_calls)

        # short-circuit: opt-out tool fired
        if any(tc["name"] == "flag_opt_out" for tc in agent_reply.tool_calls):
            outcome = "opt_out"
            break

        # agent signals end
        lower = agent_msg.lower()
        agent_closing = any(
            p in lower for p in [
                "thank you for your time",
                "you will be contacted",
                "you will receive",
                "have a good day",
                "goodbye",
            ]
        )
        if agent_closing:
            outcome = agent.classify_outcome(history, all_tool_calls)
            break

        # ---- borrower ----
        borrower_msg = borrower.reply(
            history=history,
            conversation_id=str(conversation_id),
            iteration_id=iteration_id,
        ) or "(no response)"
        history.append({"role": "user", "content": borrower_msg})
        seq += 1
        add_turn(
            conversation_id=conversation_id,
            seq=seq,
            agent_id="borrower",
            role="user",
            content=borrower_msg,
        )

    else:
        # exhausted max_turns without explicit close
        outcome = agent.classify_outcome(history, all_tool_calls)

    end_conversation(conversation_id, outcome)
    return conversation_id, AgentResult(
        outcome=outcome,
        turns=seq,
        transcript=history,
        summary_note=f"tool_calls={len(all_tool_calls)}",
        tool_calls=all_tool_calls,
    )
