"""Base class for chat-style agents (Agent 1, Agent 2 text-mode, Agent 3).

Each reply runs a *tool-use loop*: the model may emit tool_use blocks, we execute
them via ``ToolRecorder``, return tool_result blocks, and re-prompt until the
model produces a final text turn for the borrower. Only the final text becomes
a stored conversation turn; tool calls are persisted alongside the turn as
``turn.tool_calls``.

Token budget is enforced on the *agent's view* of context (system + handoff +
borrower-visible history) before the first API call. Tool-use rounds within a
single turn add to the raw API context but are not user-facing and never persist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from packages.agents.tools import ToolRecorder
from packages.llm import (
    AgentContext,
    AnthropicClient,
    LLMResponse,
)
from packages.llm.budget_tracker import budget
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
    transcript: list[dict]
    summary_note: Optional[str] = None
    tool_calls: list[dict] = field(default_factory=list)


class BaseAgent:
    agent_id: str = "base"
    model: str = DEFAULT_AGENT_MODEL
    max_tokens_out: int = 384
    temperature: float = 0.3
    tools: list[dict] = []
    max_tool_iterations: int = 6

    def __init__(
        self,
        client: Optional[AnthropicClient] = None,
        *,
        override_prompt_text: Optional[str] = None,
        override_prompt_version: Optional[int] = None,
    ) -> None:
        self.client = client or AnthropicClient()
        if override_prompt_text is not None:
            # Test/learning-loop path — skip DB lookup so we can evaluate candidate prompts.
            self.prompt_version_id = -1
            self.prompt_version_num = override_prompt_version or -1
            self.system_prompt = override_prompt_text
            from packages.llm import count_tokens
            self.system_prompt_tokens = count_tokens(override_prompt_text)
            return
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
        *,
        verification_data: Optional[dict] = None,
    ) -> AgentReply:
        """One agent turn.

        ``verification_data`` is the borrower's TRUE identity record (last4_ssn, dob,
        name). When provided, ``verify_identity`` tool calls compare against it and
        return verified=false on mismatch. When None (e.g. compliance probes with a
        ScriptedBorrower), the tool accepts everything — legacy behavior.
        """
        ctx = AgentContext(
            system_prompt=self.system_prompt,
            handoff=handoff,
            history=history,
        ).fit_to_budget()
        ctx.assert_within()
        token_counts = ctx.token_counts()

        api_messages: list[dict] = ctx.to_anthropic_messages()
        system_blocks = [{
            "type": "text",
            "text": self.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]
        recorder = ToolRecorder(verification_data=verification_data)
        total_cost = 0.0
        final_text = ""

        for _ in range(self.max_tool_iterations + 1):
            kwargs: dict = {
                "model": self.model,
                "max_tokens": self.max_tokens_out,
                "system": system_blocks,
                "messages": api_messages,
            }
            if "opus-4-7" not in self.model:
                kwargs["temperature"] = self.temperature
            if self.tools:
                kwargs["tools"] = self.tools

            resp = self.client._client.messages.create(**kwargs)
            usage = resp.usage
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
            rec = budget().record(
                provider="anthropic",
                model=self.model,
                purpose=self.agent_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
                iteration_id=iteration_id,
                conversation_id=conversation_id,
            )
            total_cost += rec.cost_usd

            # collect blocks
            text_parts: list[str] = []
            tool_use_blocks: list = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            if not tool_use_blocks:
                final_text = "\n".join(t for t in text_parts if t).strip()
                break

            # append assistant turn (text + tool_use) to messages
            assistant_content: list[dict] = []
            for t in text_parts:
                if t:
                    assistant_content.append({"type": "text", "text": t})
            for tu in tool_use_blocks:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tu.id,
                    "name": tu.name,
                    "input": tu.input,
                })
            api_messages.append({"role": "assistant", "content": assistant_content})

            # execute tools, append tool_result blocks as one user turn
            tool_results: list[dict] = []
            for tu in tool_use_blocks:
                result = recorder.handle(tu.name, tu.input or {})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(result),
                })
            api_messages.append({"role": "user", "content": tool_results})

        return AgentReply(
            text=final_text or "(no response)",
            tool_calls=[{"name": c.name, "input": c.input} for c in recorder.calls],
            token_counts=token_counts,
            cost_usd=total_cost,
        )

    # Subclasses override.
    def classify_outcome(self, transcript: list[dict], tool_calls: list[dict]) -> str:
        raise NotImplementedError
