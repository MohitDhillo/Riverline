"""Anthropic client wrapper.

Single entry point for all LLM calls in the system. Every call:
  1. Enforces the agent's token budget via AgentContext.assert_within().
  2. Uses prompt-caching on the system prompt (~90% read discount on repeat calls).
  3. Records cost + token usage to the BudgetTracker (which may raise BudgetExhausted).
  4. Returns LLMResponse with all the bookkeeping fields the eval/audit layer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic

from packages.config import settings
from packages.llm.budget_tracker import budget
from packages.llm.token_guard import AgentContext

DEFAULT_AGENT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_PROPOSER_MODEL = "claude-opus-4-7"


@dataclass
class LLMCall:
    """Inputs to a single LLM completion."""

    context: AgentContext
    purpose: str                          # 'agent_1' | 'judge' | 'simulator' | ...
    model: str = DEFAULT_AGENT_MODEL
    max_tokens: int = 512
    temperature: float = 0.3
    cache_system: bool = True
    iteration_id: Optional[int] = None
    conversation_id: Optional[str] = None
    tools: list[dict] = field(default_factory=list)
    tool_choice: Optional[dict] = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[dict]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    model: str
    purpose: str


class AnthropicClient:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or settings().anthropic_api_key
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — copy .env.example to .env and fill it in"
            )
        self._client = Anthropic(api_key=key)

    def complete(self, call: LLMCall) -> LLMResponse:
        # 1. budget guard — note: AgentContext.assert_within defaults to AGENT_BUDGET=2000.
        # That's enforced for production agents. Meta callers (summarizer, judge,
        # prompt engineer) build their AgentContext with fit_to_budget(META_BUDGET)
        # and we trust that. We still assert against META_BUDGET as a sanity backstop.
        from packages.llm.token_guard import META_BUDGET
        call.context.assert_within(agent_budget=META_BUDGET)

        # 2. build system blocks with cache_control on the system prompt
        if call.cache_system and call.context.system_prompt:
            system_blocks: list[dict] = [
                {
                    "type": "text",
                    "text": call.context.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            system_blocks = [{"type": "text", "text": call.context.system_prompt}]

        messages = call.context.to_anthropic_messages()

        kwargs: dict = {
            "model": call.model,
            "max_tokens": call.max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        # Opus 4.7 (and some other newer Anthropic models) no longer accept a `temperature`
        # parameter — the API rejects it as deprecated. Skip silently for those models.
        if "opus-4-7" not in call.model:
            kwargs["temperature"] = call.temperature
        if call.tools:
            kwargs["tools"] = call.tools
        if call.tool_choice:
            kwargs["tool_choice"] = call.tool_choice

        resp = self._client.messages.create(**kwargs)

        # 3. extract text + tool calls
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        text = "\n".join(text_parts).strip()

        # 4. usage
        usage = resp.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        # 5. record cost (may raise BudgetExhausted)
        rec = budget().record(
            provider="anthropic",
            model=call.model,
            purpose=call.purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            iteration_id=call.iteration_id,
            conversation_id=call.conversation_id,
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=rec.cost_usd,
            model=call.model,
            purpose=call.purpose,
        )
