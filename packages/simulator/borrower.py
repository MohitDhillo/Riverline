"""Borrower simulator — plays the borrower side of a conversation.

The persona's prompt template is loaded from prompts/simulator/ via the prompt
store (so personas are versioned too — they affect eval results, so they must
be reproducible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Optional

from packages.llm import AgentContext, AnthropicClient, LLMCall
from packages.llm.client import DEFAULT_AGENT_MODEL
from packages.storage.repos import get_active_prompt

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class BorrowerProfile:
    id: str
    persona: str
    name: str
    age: int
    debt_amount: float
    last4_ssn: str
    dob: str
    employment: str
    monthly_income: str
    hardship: Optional[str]
    phone: str

    def as_template_vars(self) -> dict:
        return {
            "name": self.name,
            "age": self.age,
            "debt_amount": f"{self.debt_amount:,.2f}",
            "last4_ssn": self.last4_ssn,
            "dob": self.dob,
            "employment": self.employment.replace("_", " "),
            "monthly_income": self.monthly_income,
            "hardship": self.hardship or "none",
        }


def load_borrowers(persona: Optional[str] = None) -> list[BorrowerProfile]:
    seeds = json.loads((DATA_DIR / "seeds.json").read_text())
    out = []
    for b in seeds["borrowers"]:
        if persona and b["persona"] != persona:
            continue
        out.append(BorrowerProfile(**b))
    return out


class BorrowerSimulator:
    """Plays the borrower side. Templated persona prompt + profile substitution."""

    def __init__(
        self,
        profile: BorrowerProfile,
        client: Optional[AnthropicClient] = None,
        model: str = DEFAULT_AGENT_MODEL,
        max_tokens_out: int = 200,
        temperature: float = 0.7,  # borrower is more variable
    ) -> None:
        self.profile = profile
        self.client = client or AnthropicClient()
        self.model = model
        self.max_tokens_out = max_tokens_out
        self.temperature = temperature

        pv = get_active_prompt(f"sim_{profile.persona}")
        self.persona_template = pv.prompt_text
        self.prompt_version_id = pv.id
        self.system_prompt = Template(self.persona_template).safe_substitute(
            profile.as_template_vars()
        )

    def reply(
        self,
        history: list[dict],
        conversation_id: Optional[str] = None,
        iteration_id: Optional[int] = None,
    ) -> str:
        """Note: borrower 'history' is flipped — agent's words are 'user' from borrower POV."""
        flipped = [
            {"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
            for m in history
        ]
        # If conversation just started (no prior turns from agent), seed with a generic
        # "incoming call" prompt so the model has something to respond to.
        if not flipped:
            flipped = [{"role": "user", "content": "[An AI debt collector has just messaged you. Wait for their first message before responding.]"}]

        ctx = AgentContext(
            system_prompt=self.system_prompt,
            handoff="",
            history=flipped,
        ).fit_to_budget()
        ctx.assert_within()

        resp = self.client.complete(LLMCall(
            context=ctx,
            purpose=f"sim_{self.profile.persona}",
            model=self.model,
            max_tokens=self.max_tokens_out,
            temperature=self.temperature,
            conversation_id=conversation_id,
            iteration_id=iteration_id,
        ))
        return resp.text
