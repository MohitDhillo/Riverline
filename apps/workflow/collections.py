"""CollectionsWorkflow — one workflow per borrower.

Day 1: Assessment only. Day 2 adds the rest of the pipeline:
  Assessment → Summarizer → Resolution → Summarizer → Final Notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from apps.workflow.activities import (
        ChatAgentInput,
        ChatAgentOutput,
        run_chat_agent,
    )


class Outcome(str, Enum):
    ASSESSED = "assessed"
    PARTIAL = "partial"
    NO_RESPONSE = "no_response"
    RESOLVED_AT_RESOLUTION = "resolved_at_resolution"
    RESOLVED_AT_FINAL = "resolved_at_final"
    UNRESOLVED = "unresolved"


@dataclass
class CollectionsInput:
    borrower_id: str
    iteration_id: int | None = None


@dataclass
class CollectionsOutput:
    outcome: str
    assessment_conversation_id: str
    transcript_excerpt: str


@workflow.defn
class CollectionsWorkflow:
    @workflow.run
    async def run(self, inp: CollectionsInput) -> CollectionsOutput:
        # ---- ASSESSMENT ----
        # Day 1: a single attempt. Day 2 brings the 3-retry loop.
        assess: ChatAgentOutput = await workflow.execute_activity(
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

        excerpt = "\n".join(
            f"[{t['role']}] {t['content'][:120]}"
            for t in assess.transcript[:6]
        )
        return CollectionsOutput(
            outcome=assess.outcome,
            assessment_conversation_id=assess.conversation_id,
            transcript_excerpt=excerpt,
        )
