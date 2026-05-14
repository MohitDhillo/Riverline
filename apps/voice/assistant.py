"""Vapi assistant configuration for Agent 2 (Resolution, voice).

We assemble Vapi's assistant config from:
  - Agent 2's active system prompt (DB)
  - The handoff JSON payload from Agent 1's summarizer (≤500 tokens)
  - Compliance-mandated firstMessage (AI disclosure + recording disclosure)

Vapi will then drive the LLM via its own backend; we pass the model name
and the resolved system prompt. The borrower hears Agent 2's voice via Vapi's
TTS, speaks naturally, Vapi transcribes, and at call end Vapi POSTs a transcript
webhook to ``$PUBLIC_WEBHOOK_URL/voice/callback``.
"""

from __future__ import annotations

from typing import Optional

from packages.storage.repos import get_active_prompt


def build_assistant_config(
    handoff_json: str,
    *,
    borrower_name: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
    voice: str = "Harry",
    first_message: Optional[str] = None,
) -> dict:
    """Build the assistant override payload for one outbound Vapi call.

    The handoff payload is prepended to Agent 2's system prompt as the
    in-context state from Agent 1 (mirrors how `BaseAgent.reply` does it for
    text-mode runs, so behavior is identical across both modes).
    """
    pv = get_active_prompt("agent_2")
    system_prompt = pv.prompt_text + (
        "\n\n# Inbound handoff context from Agent 1 (chat assessment stage):\n"
        f"```json\n{handoff_json}\n```\n"
        "Use this. Do not re-ask anything already captured here.\n\n"
        "# Voice call control\n"
        "This call runs on Vapi. When your Riverline prompt says to call "
        "`end_conversation(...)`, say the final borrower-facing sentence first, "
        "then immediately use the Vapi `endCall` tool. Also use `endCall` if the "
        "borrower asks to stop contact, says goodbye, or clearly indicates the "
        "conversation is over."
    )

    opener = first_message or (
        "Hello, this is an AI assistant from Riverline Collections calling about "
        "your outstanding account with Cred. This call is being recorded for "
        "compliance. Do you have a few minutes to discuss your options?"
    )

    return {
        "name": "Riverline Resolution Agent (Agent 2)",
        "firstMessage": opener,
        "model": {
            "provider": "anthropic",
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ],
            "temperature": 0.3,
            "tools": [{"type": "endCall"}],
        },
        "voice": {"provider": "vapi", "voiceId": voice},
        "transcriber": {"provider": "deepgram", "model": "nova-2"},
        "endCallPhrases": [
            "goodbye",
            "thank you for your time",
            "stop calling me",
            "you will receive written confirmation",
            "you will receive a written final notice",
        ],
        "maxDurationSeconds": 300,
        "recordingEnabled": True,
        "serverMessages": ["status-update", "end-of-call-report"],
        "metadata": {
            "agent_id": "agent_2",
            "prompt_version": pv.version,
            "borrower_name": borrower_name or "borrower",
        },
    }
