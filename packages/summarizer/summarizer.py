"""Summarizer activity — turns a transcript into a HandoffPayload.

Uses Claude Haiku 4.5 with a structured-output prompt. Parses JSON, validates
against the pydantic schema, then deterministically trims to <=500 tokens.

On failure to produce valid JSON (rare with Haiku + strict prompt), retry once;
if still bad, fall back to a heuristic skeleton extracted from the transcript so
the pipeline never deadlocks on a malformed summary.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from packages.llm import AgentContext, AnthropicClient, LLMCall
from packages.llm.client import DEFAULT_AGENT_MODEL
from packages.summarizer.schema import HandoffPayload
from packages.summarizer.trim import TrimResult, trim_to_budget

SUMMARIZER_SYSTEM = """You are a compliance-aware conversation summarizer for a debt collections AI pipeline. You read a transcript and emit a single JSON object that hands off context to the next agent stage.

You must output ONLY valid JSON matching this exact schema:

{
  "identity": {"verified": bool, "method": str, "confidence": "low"|"medium"|"high"},
  "debt": {"amount_acknowledged": float|null, "borrower_disputes": bool, "dispute_basis": str|null},
  "financial_situation": {
    "employment": "full_time"|"part_time"|"self_employed"|"unemployed"|"unknown"|null,
    "monthly_income_band": str|null,
    "stated_hardship": [str],
    "ability_to_pay_lump": "yes"|"no"|"unknown"|null,
    "ability_to_pay_plan": str|null
  },
  "offers_made": [{"type": str, "borrower_response": str}],
  "objections_raised": [str],
  "emotional_state": str,
  "compliance_flags": {
    "opt_out_requested": bool,
    "hardship_program_offered": bool,
    "sensitive_disclosure": "medical"|"job_loss"|"family_emergency"|"none"|null
  },
  "open_threads": [str],
  "borrower_quotes": [str]  // at most 3, verbatim, ≤ 80 chars each
}

# Rules
1. Be terse. Total JSON must stay under 500 tokens. Drop quotes first if you must.
2. Use `borrower_disputes=true` only if the borrower explicitly disputed the amount or the debt.
3. `compliance_flags.opt_out_requested=true` only if the borrower explicitly asked to stop contact.
4. `compliance_flags.sensitive_disclosure` is the strongest signal mentioned (medical > family_emergency > job_loss > none).
5. `compliance_flags.hardship_program_offered=true` only if the AGENT explicitly offered/mentioned a hardship program.
6. `borrower_quotes` MUST be verbatim short phrases the borrower actually said. No paraphrase.
7. `open_threads` are unresolved items the next agent will need to follow up on.
8. Output ONLY the JSON object. No prose. No markdown fences."""


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    # strip optional markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _heuristic_fallback(transcript: list[dict]) -> HandoffPayload:
    """Last-resort skeleton if the LLM returns malformed JSON."""
    text = " ".join(t.get("content", "") for t in transcript).lower()
    has_optout = any(p in text for p in ["stop contact", "stop calling", "don't contact"])
    return HandoffPayload(
        emotional_state="unknown",
        compliance_flags={
            "opt_out_requested": has_optout,
            "hardship_program_offered": "hardship" in text,
            "sensitive_disclosure": "medical" if "medical" in text else None,
        },
    )


def summarize_for_handoff(
    transcript: list[dict],
    *,
    to_agent: str,
    client: Optional[AnthropicClient] = None,
    conversation_id: Optional[str] = None,
    iteration_id: Optional[int] = None,
) -> TrimResult:
    """Return a TrimResult with a HandoffPayload <=500 tokens.

    The summarizer is a SHORT call (max 600 output tokens). Cost is recorded
    under purpose="summarizer".
    """
    client = client or AnthropicClient()
    transcript_text = "\n".join(
        f"[{t.get('agent_id', t['role'])}] {t['content']}" for t in transcript
    )
    user_msg = (
        f"You are summarizing for stage: {to_agent}\n\n"
        f"TRANSCRIPT (most recent last):\n```\n{transcript_text[:6000]}\n```\n\n"
        "Emit the JSON now."
    )

    ctx = AgentContext(
        system_prompt=SUMMARIZER_SYSTEM,
        handoff="",
        history=[{"role": "user", "content": user_msg}],
    ).fit_to_budget()
    ctx.assert_within()

    resp = client.complete(LLMCall(
        context=ctx,
        purpose="summarizer",
        model=DEFAULT_AGENT_MODEL,
        max_tokens=600,
        temperature=0.0,
        conversation_id=conversation_id,
        iteration_id=iteration_id,
    ))

    obj = _extract_json(resp.text)
    if obj is None:
        # one retry with even stricter instruction
        retry_ctx = AgentContext(
            system_prompt=SUMMARIZER_SYSTEM,
            handoff="",
            history=[
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": "That was not valid JSON. Re-emit ONLY the JSON object, no prose, no fences."},
            ],
        ).fit_to_budget()
        retry_resp = client.complete(LLMCall(
            context=retry_ctx,
            purpose="summarizer",
            model=DEFAULT_AGENT_MODEL,
            max_tokens=600,
            temperature=0.0,
            conversation_id=conversation_id,
            iteration_id=iteration_id,
        ))
        obj = _extract_json(retry_resp.text)

    if obj is None:
        payload = _heuristic_fallback(transcript)
    else:
        try:
            payload = HandoffPayload.model_validate(obj)
        except Exception:
            payload = _heuristic_fallback(transcript)

    return trim_to_budget(payload)
