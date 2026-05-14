"""CollectionsWorkflow — full A1 → A2 → A3 pipeline with summarizers between stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from apps.workflow.activities import (
        ChatAgentInput,
        ChatAgentOutput,
        SummarizeInput,
        SummarizeOutput,
        run_chat_agent,
        summarize_handoff,
    )


class Outcome(str, Enum):
    RESOLVED_AT_RESOLUTION = "resolved_at_resolution"
    RESOLVED_AT_FINAL = "resolved_at_final"
    UNRESOLVED = "unresolved"
    OPT_OUT = "opt_out"


@dataclass
class CollectionsInput:
    borrower_id: str
    iteration_id: Optional[int] = None
    max_assessment_attempts: int = 3   # spec: retry up to 3 times on no_response


@dataclass
class CollectionsOutput:
    outcome: str
    assessment_conversation_id: Optional[str] = None
    resolution_conversation_id: Optional[str] = None
    final_conversation_id: Optional[str] = None
    handoff_1_to_2_tokens: int = 0
    handoff_2_to_3_tokens: int = 0
    assessment_attempts: int = 1
    assessment_exhausted: bool = False
    summary: str = ""
    excerpts: dict = field(default_factory=dict)


def _excerpt(turns: list[dict], n: int = 6) -> str:
    return "\n".join(f"[{t['role']}] {t['content'][:120]}" for t in turns[:n])


@workflow.defn
class CollectionsWorkflow:
    @workflow.run
    async def run(self, inp: CollectionsInput) -> CollectionsOutput:
        out = CollectionsOutput(outcome=Outcome.UNRESOLVED)

        # ---- ASSESSMENT (spec: retry up to 3 attempts on no_response) ----
        t1: Optional[ChatAgentOutput] = None
        for attempt in range(1, inp.max_assessment_attempts + 1):
            t1 = await workflow.execute_activity(
                run_chat_agent,
                ChatAgentInput(
                    borrower_id=inp.borrower_id,
                    agent_id="agent_1",
                    handoff="",
                    iteration_id=inp.iteration_id,
                    max_turns=12,
                ),
                start_to_close_timeout=timedelta(minutes=10),
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            out.assessment_attempts = attempt
            if t1.outcome in ("assessed", "partial", "opt_out"):
                break
            # else: outcome was 'no_response' — retry until exhausted
        # If all retries exhausted with still no_response, spec says "proceed to
        # Resolution anyway" (the diamond's 'exhausted' edge leads to Resolution).
        if t1 is None:
            raise RuntimeError("assessment loop produced no result")
        if t1.outcome == "no_response":
            out.assessment_exhausted = True

        out.assessment_conversation_id = t1.conversation_id
        out.excerpts["agent_1"] = _excerpt(t1.transcript)
        if t1.outcome == "opt_out":
            out.outcome = Outcome.OPT_OUT
            return out

        # ---- Handoff 1→2 ----
        h2: SummarizeOutput = await workflow.execute_activity(
            summarize_handoff,
            SummarizeInput(
                conversation_ids=[t1.conversation_id],
                to_agent="to_agent_2",
                iteration_id=inp.iteration_id,
            ),
            start_to_close_timeout=timedelta(minutes=3),
            heartbeat_timeout=timedelta(minutes=1),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        out.handoff_1_to_2_tokens = h2.payload_tokens

        # ---- RESOLUTION (text-mode for Day 2; Vapi swap-in on Day 5) ----
        t2: ChatAgentOutput = await workflow.execute_activity(
            run_chat_agent,
            ChatAgentInput(
                borrower_id=inp.borrower_id,
                agent_id="agent_2",
                handoff=h2.payload_json,
                iteration_id=inp.iteration_id,
                max_turns=12,
            ),
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        out.resolution_conversation_id = t2.conversation_id
        out.excerpts["agent_2"] = _excerpt(t2.transcript)
        if t2.outcome == "deal_agreed":
            out.outcome = Outcome.RESOLVED_AT_RESOLUTION
            return out
        if t2.outcome == "opt_out":
            out.outcome = Outcome.OPT_OUT
            return out

        # ---- Handoff 2→3 (full history, both stages) ----
        h3: SummarizeOutput = await workflow.execute_activity(
            summarize_handoff,
            SummarizeInput(
                conversation_ids=[t1.conversation_id, t2.conversation_id],
                to_agent="to_agent_3",
                iteration_id=inp.iteration_id,
            ),
            start_to_close_timeout=timedelta(minutes=3),
            heartbeat_timeout=timedelta(minutes=1),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        out.handoff_2_to_3_tokens = h3.payload_tokens

        # ---- FINAL NOTICE ----
        t3: ChatAgentOutput = await workflow.execute_activity(
            run_chat_agent,
            ChatAgentInput(
                borrower_id=inp.borrower_id,
                agent_id="agent_3",
                handoff=h3.payload_json,
                iteration_id=inp.iteration_id,
                max_turns=10,
            ),
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        out.final_conversation_id = t3.conversation_id
        out.excerpts["agent_3"] = _excerpt(t3.transcript)
        if t3.outcome == "resolved":
            out.outcome = Outcome.RESOLVED_AT_FINAL
        elif t3.outcome == "opt_out":
            out.outcome = Outcome.OPT_OUT
        else:
            out.outcome = Outcome.UNRESOLVED
        return out
