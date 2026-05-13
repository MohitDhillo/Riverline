"""Rubric judge — Sonnet, optional add-on to objective metrics.

For Day 3 we DO NOT use the rubric judge in the primary metric (kept objective
+ noise-free for the gate). The judge fills in rubric scores that the writeup
and the meta-evaluator will use.

Compliance score here is a SINGLE vague 1-5 — this is the lenient v0 design
that the Day-4 meta-eval is expected to catch and replace with a per-rule
checklist judge.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from packages.llm import AgentContext, AnthropicClient, LLMCall, META_BUDGET
from packages.llm.client import DEFAULT_JUDGE_MODEL
from packages.storage.repos import get_active_prompt


def _judge_prompt() -> str:
    return get_active_prompt("judge").prompt_text


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def judge_conversation(
    agent_id: str,
    transcript: list[dict],
    *,
    client: Optional[AnthropicClient] = None,
    conversation_id: Optional[str] = None,
    iteration_id: Optional[int] = None,
) -> Optional[dict]:
    """Return parsed JSON from the rubric judge, or None if it fails."""
    client = client or AnthropicClient()
    transcript_text = "\n".join(f"[{t['role']}] {t['content']}" for t in transcript)
    user = (
        f"AGENT BEING JUDGED: {agent_id}\n\n"
        f"TRANSCRIPT:\n```\n{transcript_text[:6000]}\n```\n\n"
        "Output ONLY the JSON object."
    )
    ctx = AgentContext(
        system_prompt=_judge_prompt(),
        handoff="",
        history=[{"role": "user", "content": user}],
    ).fit_to_budget(META_BUDGET)
    ctx.assert_within(agent_budget=META_BUDGET)
    resp = client.complete(LLMCall(
        context=ctx,
        purpose="rubric_judge",
        model=DEFAULT_JUDGE_MODEL,
        max_tokens=300,
        temperature=0.0,
        conversation_id=conversation_id,
        iteration_id=iteration_id,
    ))
    return _extract_json(resp.text)
