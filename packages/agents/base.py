"""Base class for chat-style agents (Agent 1 and Agent 3).

Agent 2 (voice) shares the prompt + outcome logic but has its own runner; see
packages/agents/agent_2.py and apps/voice/.

A BaseAgent is *stateless* in itself — the caller (the conversation runner) feeds
in history and gets back one reply. Persistence happens in the runner / activity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from packages.llm import (
    AgentContext,
    AnthropicClient,
    LLMCall,
    LLMResponse,
)
from packages.llm.client import DEFAULT_AGENT_MODEL
from packages.storage.repos import get_active_prompt


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    token_counts: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    llm_response: Optional[LLMResponse] = None


@dataclass
class AgentResult:
    outcome: str
    turns: int
    transcript: list[dict]   # [{role, content}]
    summary_note: Optional[str] = None


class BaseAgent:
    """Single-agent turn driver. Loads its prompt from the DB at construction time."""

    agent_id: str = "base"
    model: str = DEFAULT_AGENT_MODEL
    max_tokens_out: int = 384
    temperature: float = 0.3

    def __init__(self, client: Optional[AnthropicClient] = None) -> None:
        self.client = client or AnthropicClient()
        pv = get_active_prompt(self.agent_id)
        self.prompt_version_id = pv.id
        self.prompt_version_num = pv.version
        self.system_prompt = pv.prompt_text
        self.system_prompt_tokens = pv.prompt_tokens

    def reply(
        self,
        history: list[dict],
        handoff: str = "",
        conversation_id: Optional[str] = None,
        iteration_id: Optional[int] = None,
    ) -> AgentReply:
        """One turn. history is the conversation so far (most recent last)."""
        ctx = AgentContext(
            system_prompt=self.system_prompt,
            handoff=handoff,
            history=history,
        )
        ctx = ctx.fit_to_budget()
        ctx.assert_within()

        resp = self.client.complete(LLMCall(
            context=ctx,
            purpose=self.agent_id,
            model=self.model,
            max_tokens=self.max_tokens_out,
            temperature=self.temperature,
            conversation_id=conversation_id,
            iteration_id=iteration_id,
        ))

        return AgentReply(
            text=resp.text,
            tool_calls=resp.tool_calls,
            token_counts=ctx.token_counts(),
            cost_usd=resp.cost_usd,
            llm_response=resp,
        )

    # Subclasses override.
    def classify_outcome(self, transcript: list[dict]) -> str:
        raise NotImplementedError
