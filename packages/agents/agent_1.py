"""Agent 1 — Assessment (chat). Outcome driven by tool calls."""

from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.agents.tools import AGENT_1_TOOLS


class AssessmentAgent(BaseAgent):
    agent_id = "agent_1"
    max_tokens_out = 384
    tools = AGENT_1_TOOLS

    def classify_outcome(self, transcript: list[dict], tool_calls: list[dict]) -> str:
        names = {tc["name"] for tc in tool_calls}
        if "flag_opt_out" in names:
            return "opt_out"

        verified = "verify_identity" in names
        disclosures = sum(1 for tc in tool_calls if tc["name"] == "record_disclosure")
        # 'assessed' = identity + 3+ disclosures captured
        if verified and disclosures >= 3:
            return "assessed"
        if verified or disclosures >= 1:
            return "partial"
        if not transcript or len([t for t in transcript if t["role"] == "user"]) == 0:
            return "no_response"
        return "partial"
