"""Agent 2 — Resolution (voice). Text-mode runner for the learning loop.

Voice transport (Vapi) lives in apps/voice/ and uses the same prompt + outcome
classifier as this text-mode runner. Per FINAL_PLAN.md the text-mode path is
kept even after Day 5 — it's how we iterate on Agent 2's prompt cheaply.
"""

from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.agents.tools import AGENT_2_TOOLS


class ResolutionAgent(BaseAgent):
    agent_id = "agent_2"
    max_tokens_out = 384
    tools = AGENT_2_TOOLS

    def classify_outcome(self, transcript: list[dict], tool_calls: list[dict]) -> str:
        names = {tc["name"] for tc in tool_calls}
        if "flag_opt_out" in names:
            return "opt_out"
        if "record_commitment" in names:
            return "deal_agreed"
        offers = [tc for tc in tool_calls if tc["name"] == "present_offer"]
        if any(o["input"].get("offer_type") == "hardship_referral" for o in offers):
            # if hardship was offered and no commitment, still consider it routed
            return "escalate_hardship"
        return "no_deal"
