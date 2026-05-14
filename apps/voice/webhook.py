"""Vapi webhook handler.

Vapi POSTs events to our `serverUrl` during/after calls. The events we care about:
  - `end-of-call-report` — fires when the call ends; payload includes transcript,
    structured tool calls, recordingUrl, summary, and the assistantOverride metadata
    (which carries our agent_id + prompt_version so we can write to the right tables).
  - `status-update` — call lifecycle (ringing/in-progress/ended); we log only.

For Day 5 we persist the end-of-call transcript into the same `conversations` +
`turns` tables that the text-mode pipeline uses, so the downstream summarizer
(handoff 2→3) and the evaluator work identically.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request

from packages.storage.repos import add_turn, create_conversation, end_conversation

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/voice/callback")
async def vapi_callback(req: Request) -> dict:
    payload = await req.json()
    msg = payload.get("message") or payload
    event_type = msg.get("type", "unknown")

    if event_type == "end-of-call-report":
        return _handle_end_of_call(msg)
    if event_type == "status-update":
        log.info("vapi status: call=%s status=%s",
                 msg.get("call", {}).get("id"), msg.get("status"))
        return {"ok": True, "ignored": event_type}
    log.info("vapi event %s (ignored on Day 5)", event_type)
    return {"ok": True, "ignored": event_type}


def _handle_end_of_call(msg: dict) -> dict:
    """Persist the transcript + recording URL into our DB."""
    call = msg.get("call", {}) or {}
    artifact = msg.get("artifact", {}) or {}
    metadata = (msg.get("assistant") or {}).get("metadata", {}) or call.get("assistant", {}).get("metadata", {})
    agent_id = metadata.get("agent_id", "agent_2")
    borrower_name = metadata.get("borrower_name", "unknown")

    conv_id = create_conversation(
        borrower_id=uuid.uuid4(),   # placeholder when the run wasn't tied to a seeded borrower
        workflow_id=call.get("id"),
        persona=f"vapi_{borrower_name}",
        agent_versions={agent_id: metadata.get("prompt_version", 1)},
    )

    # Vapi gives both a structured `messages` array (per-turn) and a flat transcript.
    # We prefer the structured one.
    seq = 0
    for m in (artifact.get("messages") or msg.get("messages") or []):
        role = m.get("role", "assistant")
        # Map Vapi's roles to ours: 'bot' -> assistant, 'user' -> user, 'system' skipped.
        if role == "system":
            continue
        if role == "bot":
            role = "assistant"
        content = m.get("message") or m.get("content") or ""
        if not content:
            continue
        seq += 1
        add_turn(
            conversation_id=conv_id,
            seq=seq,
            agent_id=agent_id if role == "assistant" else "borrower",
            role=role,
            content=content,
        )

    outcome = _classify_call_outcome(artifact.get("messages") or [])
    end_conversation(conv_id, outcome)

    log.info(
        "vapi end-of-call: call_id=%s conv=%s outcome=%s recording=%s",
        call.get("id"), conv_id, outcome, artifact.get("recordingUrl"),
    )
    return {
        "ok": True,
        "conversation_id": str(conv_id),
        "outcome": outcome,
        "recording_url": artifact.get("recordingUrl"),
        "received_at": datetime.utcnow().isoformat(),
    }


def _classify_call_outcome(messages: list[dict]) -> str:
    """Heuristic outcome classification for Day 5. Day 4-style tool calls would be
    cleaner but Vapi's tool-call surfacing varies by provider config; this works
    for the audio deliverable."""
    text = " ".join((m.get("message") or m.get("content") or "") for m in messages).lower()
    if "stop calling" in text or "do not contact" in text:
        return "opt_out"
    if "agree" in text and ("plan" in text or "monthly" in text or "settlement" in text):
        return "deal_agreed"
    if "hardship" in text and "referral" in text:
        return "escalate_hardship"
    return "no_deal"
