"""Conversation runner — pairs a BaseAgent with a BorrowerSimulator, persists turns.

The agent speaks first. Loop until max_turns or the agent's outcome classifier
says we're done. Tool calls produced inside the agent are persisted alongside
the assistant turn that emitted them.

Closing detection is gated on >=1 borrower turn — otherwise Agent 3's opening
"You will receive..." message (rule 2 boilerplate) would close the conversation
before the borrower ever speaks. Tool-driven closes (flag_opt_out, flag_for_legal,
flag_for_writeoff) are honored immediately.
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

# Tool calls that mean "we are done with this conversation, regardless of phase."
_HARD_CLOSE_TOOLS = {
    "flag_opt_out",
    "flag_for_legal",
    "flag_for_writeoff",
    "end_conversation",
}

# Phrases that *suggest* the agent is closing. We only honor these after the
# borrower has had at least one chance to reply — otherwise Agent 3's opener
# (which legitimately contains "you will receive...") would short-circuit the run.
_CLOSING_HINTS = [
    "thank you for your time",
    "you will be contacted",
    "you will receive",
    "have a good day",
    "goodbye",
    "conversation is now closed",
    "conversation has ended",
    "i cannot verify your identity",
    "i will not proceed further",
]


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

    # Pull verification data off the borrower's profile so verify_identity can do
    # a real check. For ScriptedBorrower (compliance probes) this is a placeholder
    # profile and verification_data will be None → legacy accept.
    verification_data = None
    profile = getattr(borrower, "profile", None)
    if profile is not None and getattr(profile, "last4_ssn", None):
        verification_data = {
            "last4_ssn": profile.last4_ssn,
            "dob": profile.dob,
            "name": getattr(profile, "name", None),
        }

    history: list[dict] = []
    seq = 0
    all_tool_calls: list[dict] = []
    outcome = "no_response"
    borrower_turns_taken = 0

    for _turn_idx in range(max_turns):
        # ---- agent ----
        agent_reply = agent.reply(
            history=history,
            handoff=handoff,
            conversation_id=str(conversation_id),
            iteration_id=iteration_id,
            verification_data=verification_data,
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

        # Hard close: any of the terminal tools fire → done immediately.
        fired_hard_close = next(
            (tc for tc in agent_reply.tool_calls if tc["name"] in _HARD_CLOSE_TOOLS),
            None,
        )
        if fired_hard_close:
            name = fired_hard_close["name"]
            if name == "flag_opt_out":
                outcome = "opt_out"
            elif name == "end_conversation":
                # Map the tool's `reason` to a recognized outcome where possible.
                reason = (fired_hard_close.get("input") or {}).get("reason", "other")
                if reason in ("deal_agreed", "no_resolution", "task_complete",
                               "identity_unverified", "borrower_unreachable"):
                    outcome = reason if reason != "task_complete" else \
                              agent.classify_outcome(history, all_tool_calls)
                else:
                    outcome = agent.classify_outcome(history, all_tool_calls)
            else:  # flag_for_legal / flag_for_writeoff
                outcome = agent.classify_outcome(history, all_tool_calls)
            break

        # Soft close: only honor closing phrases AFTER the borrower has spoken
        # at least once. Prevents Agent 3 from closing on its own opener.
        if borrower_turns_taken >= 1:
            lower = agent_msg.lower()
            if any(p in lower for p in _CLOSING_HINTS):
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
        borrower_turns_taken += 1
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
