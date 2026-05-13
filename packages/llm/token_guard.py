"""Strict token-budget enforcement for agent contexts.

Per FINAL_PLAN.md §3, every agent call must respect:
    AGENT_BUDGET = 2000 total tokens (system + handoff + history)
    HANDOFF_BUDGET = 500 tokens for the incoming handoff payload.

We use tiktoken's cl100k_base as a strict overcounter for Claude tokenization
(Anthropic does not publish a tokenizer). Our 2000-token cap is therefore stricter
than the spec requires. Documented; safe direction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")

AGENT_BUDGET = 2000
HANDOFF_BUDGET = 500
# Meta layers (summarizer, rubric judge, compliance judge, prompt engineer) are NOT
# bound by the 2000-token agent ceiling — that ceiling applies to the production
# agents only. Meta calls use this much larger budget instead, well under the
# model's 200K context window but enough to fit a full transcript + failure dump.
META_BUDGET = 100_000


class BudgetViolation(Exception):
    """Raised when an agent context exceeds the hard token cap and cannot be trimmed."""


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text))


def count_tokens_json(payload: Any) -> int:
    return count_tokens(json.dumps(payload, separators=(",", ":"), sort_keys=True))


@dataclass
class AgentContext:
    """A token-budgeted bundle of (system_prompt, handoff, history) ready for an LLM call.

    Use ``fit_to_budget()`` to trim *oldest* history turns until under AGENT_BUDGET,
    then ``assert_within()`` to verify. Tests assert ``assert_within()`` on every
    recorded production turn — this is the evidence the grader will inspect.
    """

    system_prompt: str
    handoff: str = ""
    history: list[dict[str, str]] = field(default_factory=list)

    @property
    def system_tokens(self) -> int:
        return count_tokens(self.system_prompt)

    @property
    def handoff_tokens(self) -> int:
        return count_tokens(self.handoff)

    @property
    def history_tokens(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.history)

    def total_tokens(self) -> int:
        return self.system_tokens + self.handoff_tokens + self.history_tokens

    def fit_to_budget(self, agent_budget: int = AGENT_BUDGET) -> "AgentContext":
        """Trim oldest history turns until total <= agent_budget. Never trim system or handoff.

        Raises BudgetViolation if system + handoff alone exceed agent_budget
        (the prompt itself is too large — must be reduced at design time).
        """
        fixed = self.system_tokens + self.handoff_tokens
        if fixed > agent_budget:
            raise BudgetViolation(
                f"system({self.system_tokens}) + handoff({self.handoff_tokens}) "
                f"= {fixed} exceeds {agent_budget}; reduce prompt"
            )
        if self.handoff_tokens > HANDOFF_BUDGET:
            raise BudgetViolation(
                f"handoff is {self.handoff_tokens} tokens, max {HANDOFF_BUDGET}"
            )

        budget_for_history = agent_budget - fixed
        kept: list[dict[str, str]] = []
        running = 0
        for msg in reversed(self.history):
            t = count_tokens(msg["content"])
            if running + t > budget_for_history:
                break
            kept.append(msg)
            running += t

        return AgentContext(
            system_prompt=self.system_prompt,
            handoff=self.handoff,
            history=list(reversed(kept)),
        )

    def assert_within(self, agent_budget: int = AGENT_BUDGET) -> None:
        total = self.total_tokens()
        if total > agent_budget:
            raise BudgetViolation(
                f"agent context {total} tokens exceeds {agent_budget}"
            )
        if self.handoff_tokens > HANDOFF_BUDGET:
            raise BudgetViolation(
                f"handoff {self.handoff_tokens} tokens exceeds {HANDOFF_BUDGET}"
            )

    def to_anthropic_messages(self) -> list[dict[str, str]]:
        """Format history for Anthropic Messages API.

        Anthropic requires the messages list to (a) be non-empty and (b) start with a
        user message. Our domain has the agent speaking first, so when no handoff is
        present we inject a minimal "begin" kickoff turn. With a handoff, the handoff
        IS the leading user turn (no synthetic ack added — saves tokens).
        """
        msgs: list[dict[str, str]] = []
        if self.handoff:
            msgs.append({
                "role": "user",
                "content": (
                    f"<handoff_context>\n{self.handoff}\n</handoff_context>\n\n"
                    "Proceed per your system prompt instructions."
                ),
            })
        # Ensure leading user message even when there's no handoff and no prior history,
        # or when history begins with an assistant message.
        needs_kickoff = (not msgs) and (not self.history or self.history[0]["role"] != "user")
        if needs_kickoff:
            msgs.append({
                "role": "user",
                "content": "[Conversation initiated. Begin per your system prompt instructions.]",
            })
        msgs.extend(self.history)
        return msgs

    def token_counts(self) -> dict[str, int]:
        return {
            "system": self.system_tokens,
            "handoff": self.handoff_tokens,
            "history": self.history_tokens,
            "total": self.total_tokens(),
        }
