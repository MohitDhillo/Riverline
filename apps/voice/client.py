"""Thin Vapi REST client.

Vapi outbound-call flow:
  1. POST /call with our transient assistant + the destination number
  2. Vapi places the call and runs Agent 2 as the assistant
  3. On call end, Vapi POSTs a 'end-of-call-report' webhook to our public URL
     with the transcript, recordingUrl, and structured tool calls.

We do NOT use a pre-configured assistantId — every Riverline call is its own
ephemeral assistant with the handoff baked into the system prompt. That way
prompt updates from the learning loop take effect on the next call without
any Vapi-dashboard ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

from apps.voice.assistant import build_assistant_config
from packages.config import settings

VAPI_BASE = "https://api.vapi.ai"


def _callback_url(url: str) -> str:
    """Accept either a base tunnel URL or the full callback route."""
    cleaned = url.rstrip("/")
    if not cleaned:
        return cleaned
    parsed = urlparse(cleaned)
    if parsed.path and parsed.path != "/":
        return cleaned
    return f"{cleaned}/voice/callback"


@dataclass
class VapiCallResult:
    call_id: str
    status: str
    phone_number_id: str
    raw: dict


class VapiClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        s = settings()
        self.api_key = api_key or s.vapi_api_key
        self.phone_number_id = s.vapi_phone_number_id
        if not self.api_key:
            raise RuntimeError(
                "VAPI_API_KEY missing in .env. Get one at https://dashboard.vapi.ai/ "
                "and re-run."
            )
        if not self.phone_number_id:
            raise RuntimeError(
                "VAPI_PHONE_NUMBER_ID missing in .env. Provision a phone number in "
                "the Vapi dashboard and copy its id here."
            )

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def start_outbound_call(
        self,
        *,
        to_number: str,
        handoff_json: str,
        borrower_name: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> VapiCallResult:
        """Place an outbound call. Returns immediately with the Vapi call id;
        the actual conversation runs asynchronously inside Vapi.
        """
        assistant = build_assistant_config(handoff_json, borrower_name=borrower_name)
        body = {
            "phoneNumberId": self.phone_number_id,
            "customer": {"number": to_number},
            "assistant": assistant,
        }
        callback = _callback_url(webhook_url or settings().public_webhook_url)
        if callback:
            body["assistant"]["serverUrl"] = callback
        with httpx.Client(timeout=30.0) as c:
            r = c.post(f"{VAPI_BASE}/call", headers=self._headers(), json=body)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Vapi call creation failed: {r.status_code} {r.text}"
                ) from e
            data = r.json()
        return VapiCallResult(
            call_id=data.get("id", ""),
            status=data.get("status", "unknown"),
            phone_number_id=self.phone_number_id,
            raw=data,
        )

    def get_call(self, call_id: str) -> dict:
        with httpx.Client(timeout=30.0) as c:
            r = c.get(f"{VAPI_BASE}/call/{call_id}", headers=self._headers())
            r.raise_for_status()
            return r.json()
