"""System-level judge — evaluates the full multi-stage pipeline transcript.

Primary metric: handoff_seamlessness (1-5).
  Did A2 re-ask anything A1 already captured? Did A3 contradict A2's offers?
  Did the borrower repeat themselves? Lower = more seams.

Used by the learning loop's system-level non-inferiority check and by the
meta-evaluator's metric-outcome correlation audit.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from packages.llm import AgentContext, AnthropicClient, LLMCall, META_BUDGET
from packages.llm.client import DEFAULT_JUDGE_MODEL

_SYSTEM = """You are a system-level judge for a multi-stage AI collections pipeline (Agent 1 chat -> summarizer -> Agent 2 voice -> summarizer -> Agent 3 chat). Given the full borrower-side experience, score how SEAMLESS the cross-modal experience was. Output strict JSON only.

Output shape:
```json
{
  "handoff_seamlessness": 1-5,
  "evidence": "one short sentence",
  "next_agent_redundant_questions": 0,
  "borrower_repeated_info_count": 0
}
```

Rubric for handoff_seamlessness (be strict):
  5 = no seams; later agents reference earlier specifics; borrower never repeats themselves
  4 = minor friction (one redundant question)
  3 = noticeable seams
  2 = jarring transitions; multiple re-asks
  1 = the borrower clearly felt handed off / had to re-introduce themselves"""


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


def judge_full_pipeline(
    transcripts_per_stage: list[tuple[str, list[dict]]],
    *,
    client: Optional[AnthropicClient] = None,
    conversation_id: Optional[str] = None,
    iteration_id: Optional[int] = None,
) -> Optional[dict]:
    """Judge a multi-stage pipeline. ``transcripts_per_stage`` is e.g.
        [("agent_1", [...]), ("agent_2", [...]), ("agent_3", [...])]
    """
    sections: list[str] = []
    for stage, turns in transcripts_per_stage:
        body = "\n".join(f"[{t['role']}] {t['content']}" for t in turns)
        sections.append(f"<stage name=\"{stage}\">\n{body}\n</stage>")
    user = (
        "Judge the seamlessness of this multi-stage pipeline.\n\n"
        + "\n".join(sections)
        + "\n\nOutput ONLY the JSON object."
    )
    client = client or AnthropicClient()
    ctx = AgentContext(
        system_prompt=_SYSTEM,
        handoff="",
        history=[{"role": "user", "content": user}],
    ).fit_to_budget(META_BUDGET)
    ctx.assert_within(agent_budget=META_BUDGET)
    resp = client.complete(LLMCall(
        context=ctx,
        purpose="system_judge",
        model=DEFAULT_JUDGE_MODEL,
        max_tokens=200,
        temperature=0.0,
        conversation_id=conversation_id,
        iteration_id=iteration_id,
    ))
    return _extract_json(resp.text)
