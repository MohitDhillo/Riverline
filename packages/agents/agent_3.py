"""Agent 3 — Final Notice (chat). Outcome: resolved | no_resolution | opt_out."""

from __future__ import annotations

from packages.agents.base import BaseAgent
from packages.agents.tools import AGENT_3_TOOLS


class FinalNoticeAgent(BaseAgent):
    agent_id = "agent_3"
    max_tokens_out = 384
    tools = AGENT_3_TOOLS

    def classify_outcome(self, transcript: list[dict], tool_calls: list[dict]) -> str:
        names = {tc["name"] for tc in tool_calls}
        if "flag_opt_out" in names:
            return "opt_out"
        for tc in tool_calls:
            if tc["name"] == "issue_final_offer" and tc["input"].get("accepted_by_borrower"):
                return "resolved"
        if "flag_for_legal" in names or "flag_for_writeoff" in names:
            return "no_resolution"
        return "no_resolution"
